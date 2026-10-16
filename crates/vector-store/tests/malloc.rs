/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

//! A test binary of its own: glibc serves a large request from a free chunk its arena
//! already holds before the mmap threshold comes into play, so the check needs a process
//! where no other test leaves such chunks behind.

#![cfg(all(target_os = "linux", target_env = "gnu"))]

use std::hint::black_box;

/// How many blocks glibc currently serves with a mapping of their own (`hblks` is a
/// process-wide count, not a per-arena one). `mallinfo2` needs glibc 2.33 or later.
fn mmapped_blocks() -> usize {
    // SAFETY: mallinfo2 only reads the allocator's statistics.
    unsafe { libc::mallinfo2() }.hblks
}

/// With the mmap threshold pinned, a 1 MiB block gets a mapping of its own and goes back to
/// the OS when freed, also after an earlier such block was freed. Without the pin, freeing
/// the first block raises the threshold above 1 MiB, and glibc carves the second one from
/// the arena, which keeps it after the free.
#[test]
fn freed_large_block_does_not_raise_mmap_threshold() {
    const SIZE: usize = 1 << 20;
    vector_store::malloc::pin_mmap_threshold().unwrap();

    // Were a 1 MiB block the thread's first allocation, it would create the thread's arena,
    // and glibc 2.43 sizes a new arena for that request: the block would come from it, pinned
    // or not (2.39 allocates its tcache first, so the arena starts small). A small block first
    // rules that out, whatever the harness allocated on this thread. Every block goes through
    // black_box: LLVM removes a malloc/free pair whose memory is never used, and the test
    // would then allocate nothing.
    // SAFETY: plain malloc/free pair; the block is not used otherwise.
    unsafe { libc::free(black_box(libc::malloc(64))) };

    for round in 1..=2 {
        let before = mmapped_blocks();
        // SAFETY: plain malloc/free pair; the block is not used otherwise.
        let block = black_box(unsafe { libc::malloc(SIZE) });
        assert!(!block.is_null());
        let mmapped = mmapped_blocks() > before;
        // SAFETY: `block` was returned by malloc above and is freed once.
        unsafe { libc::free(block) };
        assert!(mmapped, "1 MiB block of round {round} is not mmapped");
        assert_eq!(
            mmapped_blocks(),
            before,
            "1 MiB block of round {round} is not unmapped when freed"
        );
    }
}
