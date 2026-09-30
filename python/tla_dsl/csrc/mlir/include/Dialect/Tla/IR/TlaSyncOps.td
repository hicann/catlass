//===----------------------------------------------------------------------===//
// Synchronization ops: flags, barriers and mutexes.
//
// Included by Dialect/Tla/IR/Tla.td, the file mlir-tblgen is run on.
//===----------------------------------------------------------------------===//

def Tla_CrossCoreSetFlagOp : Tla_Op<"cross_core_set_flag", []> {
  let summary = "Cross-core flag set";
  let description = [{
    Set a cross-core synchronization flag.
  }];
  let arguments = (ins Tla_CrossFlagType:$flag, Tla_PipeAttr:$pipe,
                       OptionalAttr<I64Attr>:$aiv_id);
  let assemblyFormat = [{
    $flag attr-dict `:` qualified(type($flag))
  }];
  let hasVerifier = 1;
}

def Tla_CrossCoreWaitFlagOp : Tla_Op<"cross_core_wait_flag", []> {
  let summary = "Cross-core flag wait";
  let description = [{
    Wait on a cross-core synchronization flag.
  }];
  let arguments = (ins Tla_CrossFlagType:$flag, Tla_PipeAttr:$pipe,
                       OptionalAttr<I64Attr>:$aiv_id);
  let assemblyFormat = [{
    $flag attr-dict `:` qualified(type($flag))
  }];
  let hasVerifier = 1;
}

def Tla_CrossFlagOp : Tla_Op<"cross_flag", [Pure]> {
  let summary = "Cross-block flag literal";
  let description = [{
    Materialize a cross-core synchronization flag.
  }];
  let arguments = (ins StrAttr:$name);
  let results = (outs Tla_CrossFlagType:$flag);
  let assemblyFormat = [{
    $name attr-dict `->` qualified(type($flag))
  }];
}

def Tla_FlagOp : Tla_Op<"flag", [Pure]> {
  let summary = "Flag literal";
  let description = [{
    Materialize a static synchronization flag.
  }];
  let arguments = (ins StrAttr:$name, Tla_PipeAttr:$src_pipe, Tla_PipeAttr:$dst_pipe);
  let results = (outs Tla_FlagType:$flag);
  let assemblyFormat = [{
    $name attr-dict `->` qualified(type($flag))
  }];
}

def Tla_PipeBarrierOp : Tla_Op<"pipe_barrier", []> {
  let summary = "Pipe barrier";
  let description = [{
    Apply a synchronization barrier to the specified pipe.
  }];
  let arguments = (ins Tla_PipeAttr:$pipe);
  let assemblyFormat = [{
    `[` $pipe `]` attr-dict
  }];
  let hasVerifier = 1;
}

def Tla_MutexOp : Tla_Op<"mutex", [Pure]> {
  let summary = "Mutex literal";
  let description = [{
    Materialize a static mutex associated with a semantic resource.
    ``id = -1`` means the compiler may allocate the hardware MutexID later;
    ``0`` through ``31`` are user-provided hardware MutexIDs.
  }];
  let arguments = (ins StrAttr:$resource, I64Attr:$id);
  let results = (outs Tla_MutexType:$mutex);
  let assemblyFormat = [{
    $resource attr-dict `->` qualified(type($mutex))
  }];
}

def Tla_MutexLockOp : Tla_Op<"mutex_lock", []> {
  let summary = "Mutex lock";
  let description = [{
    Acquire a mutex from the specified pipe.
  }];
  let arguments = (ins Tla_MutexType:$mutex, Tla_PipeAttr:$pipe);
  let assemblyFormat = [{
    $mutex `[` $pipe `]` `:` qualified(type($mutex)) attr-dict
  }];
  let hasVerifier = 1;
}

def Tla_MutexUnlockOp : Tla_Op<"mutex_unlock", []> {
  let summary = "Mutex unlock";
  let description = [{
    Release a mutex from the specified pipe.
  }];
  let arguments = (ins Tla_MutexType:$mutex, Tla_PipeAttr:$pipe);
  let assemblyFormat = [{
    $mutex `[` $pipe `]` `:` qualified(type($mutex)) attr-dict
  }];
  let hasVerifier = 1;
}

def Tla_LocalMemBarOp : Tla_Op<"local_mem_bar", []> {
  let summary = "Local memory barrier";
  let description = [{
    Apply a synchronization barrier to the specified memory type.
    The ``barrier_kind`` is an encoded I32 immediate that selects the
    src/dst pair (see hivmave.membar for the encoding scheme).
  }];
  let arguments = (ins I64Attr:$barrier_kind);
  let assemblyFormat = "$barrier_kind attr-dict";
  let hasVerifier = 1;
}

def Tla_SetFlagOp : Tla_Op<"set_flag", []> {
  let summary = "Set flag";
  let description = [{
    Signal a synchronization flag from one pipe to another.
  }];
  let arguments = (ins Tla_FlagType:$flag);
  let assemblyFormat = [{
    $flag `:` qualified(type($flag)) attr-dict
  }];
  let hasVerifier = 1;
}

def Tla_WaitFlagOp : Tla_Op<"wait_flag", []> {
  let summary = "Wait flag";
  let description = [{
    Wait on a synchronization flag from one pipe to another.
  }];
  let arguments = (ins Tla_FlagType:$flag);
  let assemblyFormat = [{
    $flag `:` qualified(type($flag)) attr-dict
  }];
  let hasVerifier = 1;
}
