# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: real_time_payoff_tracker.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='real_time_payoff_tracker.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x1ereal_time_payoff_tracker.proto\x1a\x1bgoogle/protobuf/empty.proto\"<\n\x0ePayoffTableKey\x12\x1d\n\x15payoff_table_is_empty\x18\x01 \x01(\x08\x12\x0b\n\x03key\x18\x02 \x01(\t\"q\n\nPolicyInfo\x12\x12\n\npolicy_key\x18\x01 \x01(\t\x12\x1f\n\x17policy_model_config_key\x18\x02 \x01(\t\x12\x19\n\x11policy_class_name\x18\x03 \x01(\t\x12\x13\n\x0bpolicy_tags\x18\x04 \x03(\t\"N\n\x0eLivePolicyInfo\x12 \n\x0bpolicy_info\x18\x01 \x01(\x0b\x32\x0b.PolicyInfo\x12\x1a\n\x12payoff_table_index\x18\x02 \x01(\x05\"5\n\x0c\x43onfirmation\x12\x14\n\x0c\x63onfirmation\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t\"l\n\x11\x45valMatchupResult\x12\x15\n\ras_policy_key\x18\x01 \x01(\t\x12\x1a\n\x12\x61gainst_policy_key\x18\x02 \x01(\t\x12\x0e\n\x06payoff\x18\x03 \x01(\x02\x12\x14\n\x0cgames_played\x18\x04 \x01(\x05\x32\xec\x01\n\x15RealTimePayoffTracker\x12H\n\x1bGetLatestLivePayoffTableKey\x12\x16.google.protobuf.Empty\x1a\x0f.PayoffTableKey\"\x00\x12\x45\n!SubmitNewLivePolicyForPayoffTable\x12\x0f.LivePolicyInfo\x1a\r.Confirmation\"\x00\x12\x42\n\x1bSubmitLiveEvalMatchupResult\x12\x12.EvalMatchupResult\x1a\r.Confirmation\"\x00\x62\x06proto3'
  ,
  dependencies=[google_dot_protobuf_dot_empty__pb2.DESCRIPTOR,])




_PAYOFFTABLEKEY = _descriptor.Descriptor(
  name='PayoffTableKey',
  full_name='PayoffTableKey',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='payoff_table_is_empty', full_name='PayoffTableKey.payoff_table_is_empty', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='key', full_name='PayoffTableKey.key', index=1,
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
  serialized_start=63,
  serialized_end=123,
)


_POLICYINFO = _descriptor.Descriptor(
  name='PolicyInfo',
  full_name='PolicyInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='policy_key', full_name='PolicyInfo.policy_key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='policy_model_config_key', full_name='PolicyInfo.policy_model_config_key', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='policy_class_name', full_name='PolicyInfo.policy_class_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='policy_tags', full_name='PolicyInfo.policy_tags', index=3,
      number=4, type=9, cpp_type=9, label=3,
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
  serialized_start=125,
  serialized_end=238,
)


_LIVEPOLICYINFO = _descriptor.Descriptor(
  name='LivePolicyInfo',
  full_name='LivePolicyInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='policy_info', full_name='LivePolicyInfo.policy_info', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='payoff_table_index', full_name='LivePolicyInfo.payoff_table_index', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=240,
  serialized_end=318,
)


_CONFIRMATION = _descriptor.Descriptor(
  name='Confirmation',
  full_name='Confirmation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='confirmation', full_name='Confirmation.confirmation', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='message', full_name='Confirmation.message', index=1,
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
  serialized_start=320,
  serialized_end=373,
)


_EVALMATCHUPRESULT = _descriptor.Descriptor(
  name='EvalMatchupResult',
  full_name='EvalMatchupResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='as_policy_key', full_name='EvalMatchupResult.as_policy_key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='against_policy_key', full_name='EvalMatchupResult.against_policy_key', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='payoff', full_name='EvalMatchupResult.payoff', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='games_played', full_name='EvalMatchupResult.games_played', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=375,
  serialized_end=483,
)

_LIVEPOLICYINFO.fields_by_name['policy_info'].message_type = _POLICYINFO
DESCRIPTOR.message_types_by_name['PayoffTableKey'] = _PAYOFFTABLEKEY
DESCRIPTOR.message_types_by_name['PolicyInfo'] = _POLICYINFO
DESCRIPTOR.message_types_by_name['LivePolicyInfo'] = _LIVEPOLICYINFO
DESCRIPTOR.message_types_by_name['Confirmation'] = _CONFIRMATION
DESCRIPTOR.message_types_by_name['EvalMatchupResult'] = _EVALMATCHUPRESULT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PayoffTableKey = _reflection.GeneratedProtocolMessageType('PayoffTableKey', (_message.Message,), {
  'DESCRIPTOR' : _PAYOFFTABLEKEY,
  '__module__' : 'real_time_payoff_tracker_pb2'
  # @@protoc_insertion_point(class_scope:PayoffTableKey)
  })
_sym_db.RegisterMessage(PayoffTableKey)

PolicyInfo = _reflection.GeneratedProtocolMessageType('PolicyInfo', (_message.Message,), {
  'DESCRIPTOR' : _POLICYINFO,
  '__module__' : 'real_time_payoff_tracker_pb2'
  # @@protoc_insertion_point(class_scope:PolicyInfo)
  })
_sym_db.RegisterMessage(PolicyInfo)

LivePolicyInfo = _reflection.GeneratedProtocolMessageType('LivePolicyInfo', (_message.Message,), {
  'DESCRIPTOR' : _LIVEPOLICYINFO,
  '__module__' : 'real_time_payoff_tracker_pb2'
  # @@protoc_insertion_point(class_scope:LivePolicyInfo)
  })
_sym_db.RegisterMessage(LivePolicyInfo)

Confirmation = _reflection.GeneratedProtocolMessageType('Confirmation', (_message.Message,), {
  'DESCRIPTOR' : _CONFIRMATION,
  '__module__' : 'real_time_payoff_tracker_pb2'
  # @@protoc_insertion_point(class_scope:Confirmation)
  })
_sym_db.RegisterMessage(Confirmation)

EvalMatchupResult = _reflection.GeneratedProtocolMessageType('EvalMatchupResult', (_message.Message,), {
  'DESCRIPTOR' : _EVALMATCHUPRESULT,
  '__module__' : 'real_time_payoff_tracker_pb2'
  # @@protoc_insertion_point(class_scope:EvalMatchupResult)
  })
_sym_db.RegisterMessage(EvalMatchupResult)



_REALTIMEPAYOFFTRACKER = _descriptor.ServiceDescriptor(
  name='RealTimePayoffTracker',
  full_name='RealTimePayoffTracker',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=486,
  serialized_end=722,
  methods=[
  _descriptor.MethodDescriptor(
    name='GetLatestLivePayoffTableKey',
    full_name='RealTimePayoffTracker.GetLatestLivePayoffTableKey',
    index=0,
    containing_service=None,
    input_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    output_type=_PAYOFFTABLEKEY,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='SubmitNewLivePolicyForPayoffTable',
    full_name='RealTimePayoffTracker.SubmitNewLivePolicyForPayoffTable',
    index=1,
    containing_service=None,
    input_type=_LIVEPOLICYINFO,
    output_type=_CONFIRMATION,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='SubmitLiveEvalMatchupResult',
    full_name='RealTimePayoffTracker.SubmitLiveEvalMatchupResult',
    index=2,
    containing_service=None,
    input_type=_EVALMATCHUPRESULT,
    output_type=_CONFIRMATION,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_REALTIMEPAYOFFTRACKER)

DESCRIPTOR.services_by_name['RealTimePayoffTracker'] = _REALTIMEPAYOFFTRACKER

# @@protoc_insertion_point(module_scope)
