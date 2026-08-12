# Deferred: replace `@__allow_legacy_any_origin_fields` with real origins

**Filed:** 2026-07-24, during the `dev2026072306` nightly migration.
**Status:** deferred — escape hatch applied, redesign not started.

## What happened

Mojo `dev2026072306` made it an error for a struct field to expose `AnyOrigin`
in its type:

```
error: struct fields cannot expose AnyOrigin in their type;
       a_ptr has type 'Pointer[CachedMatrix, MutAnyOrigin, _safe=False]'
```

The rationale is that `MutAnyOrigin` silently extends unrelated lifetimes and
disables exclusivity checking. Storing it in a struct field makes that escape
durable rather than momentary — the compiler can no longer reason about how long
the pointee must outlive the struct.

To unblock the migration we applied the `@__allow_legacy_any_origin_fields`
field decorator, which is what napi-mojo itself does for `ModuleBuilder`. This
restores the old (unchecked) behavior. It is a stopgap, not a fix.

## Where the hatch is applied

| File | Struct | Fields |
|---|---|---|
| `packages/retrieve/src/kernels.mojo` | `MatmulAsyncData` | `deferred`, `work`, `a_ref`, `b_ref`, `dst_ref`, `a_ptr`, `b_ptr`, `dst_ptr`, `state_ptr` |
| `packages/retrieve/src/kernels.mojo` | `SearchAsyncData` | `deferred`, `work`, `a_ref`, `b_ref`, `idx_ref`, `scores_ref`, `a_ptr`, `b_ptr`, `idx_ptr`, `scores_ptr`, `state_ptr` |
| `packages/embed/src/embed.mojo` | `EmbedAsyncData` | `deferred`, `work`, `ids_ref`, `mask_ref`, `dst_ref` |
| `examples/stats/addon_cached.mojo` | `CachedStats` | `data_f64` |

`GpuState` and `CachedMatrix` did **not** need it — they hold owned
`DeviceBuffer`s and plain scalars.

## Why this is worth doing properly

These are exactly the structs where lifetime bugs would be worst. Each async
struct outlives the N-API callback that created it: it is heap-allocated, handed
to `napi_create_async_work`, and torn down in the complete-callback on a
different thread. The `*_ptr` fields point into V8-owned TypedArray memory whose
liveness is guaranteed only by the sibling `*_ref` `napi_ref` handles. Nothing in
the type system currently ties those two together — the invariant "`a_ptr` is
valid because `a_ref` is still alive" lives only in comments.

That is a real class of use-after-free, and it is unchecked today.

## Sketch of the fix

1. Give the async structs an explicit origin parameter tied to the ref handles,
   so `a_ptr`'s origin is derived from `a_ref` rather than erased.
2. Consider wrapping the `ref`/`ptr` pairs into a single owning type
   (`PinnedTypedArray`?) that holds the `napi_ref` and exposes the pointer only
   through a method whose return origin borrows `self`. This makes the invariant
   structural rather than documentary.
3. `NapiRef`/`NapiDeferred`/`NapiAsyncWork` are aliases for
   `OpaquePointer[MutAnyOrigin]` in napi-mojo. Fixing them properly likely needs
   an upstream napi-mojo change, so coordinate rather than patching downstream.

## Constraint

Step 3 means this cannot be finished purely inside this repo. napi-mojo's own
`ModuleBuilder` carries the same hatch, so the redesign should probably start
there.
