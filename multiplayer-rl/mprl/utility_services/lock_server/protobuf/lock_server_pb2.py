# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: lock_server.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='lock_server.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x11lock_server.proto\"z\n\x0bLockRequest\x12\x11\n\tlock_name\x18\x01 \x01(\t\x12\x11\n\tworker_id\x18\x02 \x01(\t\x12\x1f\n\x17remain_after_disconnect\x18\x03 \x01(\x08\x12$\n\x1crequest_checkpoint_with_name\x18\x04 \x01(\t\"8\n\x0eLockWorkerPing\x12\x13\n\x0bworker_type\x18\x01 \x01(\t\x12\x11\n\tworker_id\x18\x02 \x01(\t\"9\n\x10LockConfirmation\x12\x14\n\x0c\x63onfirmation\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t\"\x1e\n\x08LockList\x12\x12\n\nlock_names\x18\x01 \x03(\t\"\x1c\n\nNameFilter\x12\x0e\n\x06\x66ilter\x18\x01 \x01(\t\"\xa6\x01\n\x12LockReplaceRequest\x12\x15\n\rold_lock_name\x18\x01 \x01(\t\x12\x15\n\rnew_lock_name\x18\x02 \x01(\t\x12\x11\n\tworker_id\x18\x03 \x01(\t\x12)\n!new_lock_remains_after_disconnect\x18\x04 \x01(\x08\x12$\n\x1crequest_checkpoint_with_name\x18\x05 \x01(\t2\x90\x02\n\nLockServer\x12\x36\n\x11TryToCheckoutLock\x12\x0c.LockRequest\x1a\x11.LockConfirmation\"\x00\x12\x30\n\x0bReleaseLock\x12\x0c.LockRequest\x1a\x11.LockConfirmation\"\x00\x12\x31\n\x15GetAllLocksWithString\x12\x0b.NameFilter\x1a\t.LockList\"\x00\x12\x37\n\x0bReplaceLock\x12\x13.LockReplaceRequest\x1a\x11.LockConfirmation\"\x00\x12,\n\x04Ping\x12\x0f.LockWorkerPing\x1a\x11.LockConfirmation\"\x00\x62\x06proto3'
)




_LOCKREQUEST = _descriptor.Descriptor(
  name='LockRequest',
  full_name='LockRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='lock_name', full_name='LockRequest.lock_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='worker_id', full_name='LockRequest.worker_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='remain_after_disconnect', full_name='LockRequest.remain_after_disconnect', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='request_checkpoint_with_name', full_name='LockRequest.request_checkpoint_with_name', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=21,
  serialized_end=143,
)


_LOCKWORKERPING = _descriptor.Descriptor(
  name='LockWorkerPing',
  full_name='LockWorkerPing',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='worker_type', full_name='LockWorkerPing.worker_type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='worker_id', full_name='LockWorkerPing.worker_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=145,
  serialized_end=201,
)


_LOCKCONFIRMATION = _descriptor.Descriptor(
  name='LockConfirmation',
  full_name='LockConfirmation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='confirmation', full_name='LockConfirmation.confirmation', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='message', full_name='LockConfirmation.message', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=203,
  serialized_end=260,
)


_LOCKLIST = _descriptor.Descriptor(
  name='LockList',
  full_name='LockList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='lock_names', full_name='LockList.lock_names', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=262,
  serialized_end=292,
)


_NAMEFILTER = _descriptor.Descriptor(
  name='NameFilter',
  full_name='NameFilter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='filter', full_name='NameFilter.filter', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=294,
  serialized_end=322,
)


_LOCKREPLACEREQUEST = _descriptor.Descriptor(
  name='LockReplaceRequest',
  full_name='LockReplaceRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='old_lock_name', full_name='LockReplaceRequest.old_lock_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='new_lock_name', full_name='LockReplaceRequest.new_lock_name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='worker_id', full_name='LockReplaceRequest.worker_id', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='new_lock_remains_after_disconnect', full_name='LockReplaceRequest.new_lock_remains_after_disconnect', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='request_checkpoint_with_name', full_name='LockReplaceRequest.request_checkpoint_with_name', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=325,
  serialized_end=491,
)

DESCRIPTOR.message_types_by_name['LockRequest'] = _LOCKREQUEST
DESCRIPTOR.message_types_by_name['LockWorkerPing'] = _LOCKWORKERPING
DESCRIPTOR.message_types_by_name['LockConfirmation'] = _LOCKCONFIRMATION
DESCRIPTOR.message_types_by_name['LockList'] = _LOCKLIST
DESCRIPTOR.message_types_by_name['NameFilter'] = _NAMEFILTER
DESCRIPTOR.message_types_by_name['LockReplaceRequest'] = _LOCKREPLACEREQUEST
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

LockRequest = _reflection.GeneratedProtocolMessageType('LockRequest', (_message.Message,), {
  'DESCRIPTOR' : _LOCKREQUEST,
  '__module__' : 'lock_server_pb2'
  # @@protoc_insertion_point(class_scope:LockRequest)
  })
_sym_db.RegisterMessage(LockRequest)

LockWorkerPing = _reflection.GeneratedProtocolMessageType('LockWorkerPing', (_message.Message,), {
  'DESCRIPTOR' : _LOCKWORKERPING,
  '__module__' : 'lock_server_pb2'
  # @@protoc_insertion_point(class_scope:LockWorkerPing)
  })
_sym_db.RegisterMessage(LockWorkerPing)

LockConfirmation = _reflection.GeneratedProtocolMessageType('LockConfirmation', (_message.Message,), {
  'DESCRIPTOR' : _LOCKCONFIRMATION,
  '__module__' : 'lock_server_pb2'
  # @@protoc_insertion_point(class_scope:LockConfirmation)
  })
_sym_db.RegisterMessage(LockConfirmation)

LockList = _reflection.GeneratedProtocolMessageType('LockList', (_message.Message,), {
  'DESCRIPTOR' : _LOCKLIST,
  '__module__' : 'lock_server_pb2'
  # @@protoc_insertion_point(class_scope:LockList)
  })
_sym_db.RegisterMessage(LockList)

NameFilter = _reflection.GeneratedProtocolMessageType('NameFilter', (_message.Message,), {
  'DESCRIPTOR' : _NAMEFILTER,
  '__module__' : 'lock_server_pb2'
  # @@protoc_insertion_point(class_scope:NameFilter)
  })
_sym_db.RegisterMessage(NameFilter)

LockReplaceRequest = _reflection.GeneratedProtocolMessageType('LockReplaceRequest', (_message.Message,), {
  'DESCRIPTOR' : _LOCKREPLACEREQUEST,
  '__module__' : 'lock_server_pb2'
  # @@protoc_insertion_point(class_scope:LockReplaceRequest)
  })
_sym_db.RegisterMessage(LockReplaceRequest)



_LOCKSERVER = _descriptor.ServiceDescriptor(
  name='LockServer',
  full_name='LockServer',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=494,
  serialized_end=766,
  methods=[
  _descriptor.MethodDescriptor(
    name='TryToCheckoutLock',
    full_name='LockServer.TryToCheckoutLock',
    index=0,
    containing_service=None,
    input_type=_LOCKREQUEST,
    output_type=_LOCKCONFIRMATION,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='ReleaseLock',
    full_name='LockServer.ReleaseLock',
    index=1,
    containing_service=None,
    input_type=_LOCKREQUEST,
    output_type=_LOCKCONFIRMATION,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GetAllLocksWithString',
    full_name='LockServer.GetAllLocksWithString',
    index=2,
    containing_service=None,
    input_type=_NAMEFILTER,
    output_type=_LOCKLIST,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='ReplaceLock',
    full_name='LockServer.ReplaceLock',
    index=3,
    containing_service=None,
    input_type=_LOCKREPLACEREQUEST,
    output_type=_LOCKCONFIRMATION,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Ping',
    full_name='LockServer.Ping',
    index=4,
    containing_service=None,
    input_type=_LOCKWORKERPING,
    output_type=_LOCKCONFIRMATION,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_LOCKSERVER)

DESCRIPTOR.services_by_name['LockServer'] = _LOCKSERVER

# @@protoc_insertion_point(module_scope)