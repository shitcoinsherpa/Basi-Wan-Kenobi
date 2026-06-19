"""Rolling pin window for async block-swap.

Maintains a sliding set of K blocks whose host storage is currently
page-locked via cudaHostRegister. As blocks are about to be prefetched
(async H2D), we register them; once they're no longer needed in the
window, we unregister (evict). This keeps total pinned memory bounded
to K × block_bytes — fits within WSL2 caps even when full-pinning
(N × block_bytes ≈ 19 GB) does not.

The ring buffer does NOT allocate any storage of its own. It only
manages the page-lock state of the blocks' existing storage. Async H2D
from this storage to GPU then runs concurrently with compute, matching
the measured 28% per-step speedup but in a cap-safe way.

State machine per block:
  UNPINNED -> PINNED (when ensure_pinned called)
  PINNED -> UNPINNED (when evicted by ring rotation)

Eviction policy: FIFO. The oldest pinned block is evicted when a new
slot is needed and the ring is full. For Wan2.2 block-swap with a small
K and deterministic forward order, FIFO is optimal — we always evict
the block we just finished using.
"""
import collections


def _first_cpu_ptr(block):
    """Return the storage data_ptr of the first CPU param/buffer of `block`,
    or None if none found. Used to detect storage reallocation across
    .to() round-trips."""
    for p in block.parameters(recurse=True):
        t = p.data
        if t is not None and t.device.type == 'cpu':
            return t.untyped_storage().data_ptr()
    for b in block.buffers(recurse=True):
        t = b.data
        if t is not None and t.device.type == 'cpu':
            return t.untyped_storage().data_ptr()
    for mod in block.modules():
        if not hasattr(mod, '_basiwan_packed'):
            continue
        for name in ('weight', 'bias', 'lora_down', 'lora_up'):
            t = getattr(mod, name, None)
            if t is not None and t.device.type == 'cpu':
                return t.untyped_storage().data_ptr()
        mp = mod._basiwan_packed
        if mp is not None:
            for name in ('weight', 'weight_hi', 'scales', 'mins'):
                t = getattr(mp, name, None)
                if t is not None and t.device.type == 'cpu':
                    return t.untyped_storage().data_ptr()
    return None


class RingBuffer:
    """Tracks K blocks whose storage is currently pinned via cudaHostRegister."""

    def __init__(self, k, cudart):
        if k <= 0:
            raise ValueError(f"K must be positive, got {k}")
        self.k = k
        self.cudart = cudart
        # FIFO of currently-pinned block ids
        self.pinned_order = collections.deque()
        # block_id -> list of (ptr, nbytes) for unregistering
        self.registrations = {}

    def ensure_pinned(self, block_id, block):
        """Pin `block`'s storage if not already pinned. Evicts oldest if full.

        Re-registers if the block's current storage ptr differs from the one
        we previously pinned (e.g., after block.to('cpu') reallocates the
        host side into fresh pageable memory).

        Returns True if newly registered, False if our existing registration
        still matches the block's current storage.
        """
        cur_ptr = _first_cpu_ptr(block)
        existing = self.registrations.get(block_id)
        if existing is not None:
            existing_first_ptr = existing[0][1] if existing else None
            if existing_first_ptr == cur_ptr and cur_ptr is not None:
                return False
            # Stale — unregister it, take it out of FIFO, fall through.
            for _stor, ptr, _nb in existing:
                try:
                    self.cudart.cudaHostUnregister(ptr)
                except Exception:
                    pass
            del self.registrations[block_id]
            try:
                self.pinned_order.remove(block_id)
            except ValueError:
                pass
        if len(self.pinned_order) >= self.k:
            self._evict_oldest()
        regs = self._register_block(block)
        self.registrations[block_id] = regs
        self.pinned_order.append(block_id)
        return True

    def _evict_oldest(self):
        old_id = self.pinned_order.popleft()
        regs = self.registrations.pop(old_id, [])
        # regs items are (storage_handle, ptr, nbytes). We unregister BEFORE
        # the storage handle goes out of scope so cudaHostUnregister sees a
        # still-valid pointer.
        for _stor, ptr, _nb in regs:
            try:
                self.cudart.cudaHostUnregister(ptr)
            except Exception:
                pass
        # storage handles drop here.

    def _register_block(self, block):
        regs = []
        cudart = self.cudart
        seen_ptrs = set()

        def _try_register(t):
            if t is None or t.device.type != 'cpu' or t.is_pinned():
                return
            try:
                s = t.untyped_storage()
                ptr, nb = s.data_ptr(), s.nbytes()
                if ptr in seen_ptrs:
                    return
                rc = cudart.cudaHostRegister(ptr, nb, 0)
                if int(rc) == 0:
                    # Hold the storage handle so the underlying memory is
                    # NOT freed before we get a chance to cudaHostUnregister
                    # it (Module._apply replaces param tensors on .to(),
                    # dropping the source's last Python ref otherwise).
                    regs.append((s, ptr, nb))
                    seen_ptrs.add(ptr)
            except Exception:
                pass

        for p in block.parameters(recurse=True):
            _try_register(p.data)
        for b in block.buffers(recurse=True):
            _try_register(b.data)
        for mod in block.modules():
            if not hasattr(mod, '_basiwan_packed'):
                continue
            for name in ('weight', 'bias', 'lora_down', 'lora_up'):
                _try_register(getattr(mod, name, None))
            mp = mod._basiwan_packed
            if mp is not None:
                for name in ('weight', 'weight_hi', 'scales', 'mins'):
                    _try_register(getattr(mp, name, None))
        return regs

    def drain(self):
        """Unregister ALL pinned blocks. Called at MoE boundary."""
        while self.pinned_order:
            self._evict_oldest()

    def release(self):
        """Release all registrations on shutdown."""
        self.drain()


def measure_block_bytes(block):
    """Walk all leaf tensors in `block` and sum their byte counts."""
    total = 0
    for p in block.parameters(recurse=True):
        total += p.untyped_storage().nbytes()
    for b in block.buffers(recurse=True):
        total += b.untyped_storage().nbytes()
    for mod in block.modules():
        if hasattr(mod, '_basiwan_packed'):
            for name in ('weight', 'bias', 'lora_down', 'lora_up'):
                t = getattr(mod, name, None)
                if t is not None:
                    total += t.untyped_storage().nbytes()
            mp = mod._basiwan_packed
            if mp is not None:
                for name in ('weight', 'weight_hi', 'scales', 'mins'):
                    t = getattr(mp, name, None)
                    if t is not None:
                        total += t.untyped_storage().nbytes()
    return total
