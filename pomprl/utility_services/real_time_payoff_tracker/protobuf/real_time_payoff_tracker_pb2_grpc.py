# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
import real_time_payoff_tracker_pb2 as real__time__payoff__tracker__pb2


class RealTimePayoffTrackerStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.GetLatestLivePayoffTableKey = channel.unary_unary(
        '/RealTimePayoffTracker/GetLatestLivePayoffTableKey',
        request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
        response_deserializer=real__time__payoff__tracker__pb2.PayoffTableKey.FromString,
        )
    self.SubmitNewLivePolicyForPayoffTable = channel.unary_unary(
        '/RealTimePayoffTracker/SubmitNewLivePolicyForPayoffTable',
        request_serializer=real__time__payoff__tracker__pb2.LivePolicyInfo.SerializeToString,
        response_deserializer=real__time__payoff__tracker__pb2.Confirmation.FromString,
        )
    self.SubmitLiveEvalMatchupResult = channel.unary_unary(
        '/RealTimePayoffTracker/SubmitLiveEvalMatchupResult',
        request_serializer=real__time__payoff__tracker__pb2.EvalMatchupResult.SerializeToString,
        response_deserializer=real__time__payoff__tracker__pb2.Confirmation.FromString,
        )


class RealTimePayoffTrackerServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def GetLatestLivePayoffTableKey(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def SubmitNewLivePolicyForPayoffTable(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def SubmitLiveEvalMatchupResult(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_RealTimePayoffTrackerServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'GetLatestLivePayoffTableKey': grpc.unary_unary_rpc_method_handler(
          servicer.GetLatestLivePayoffTableKey,
          request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
          response_serializer=real__time__payoff__tracker__pb2.PayoffTableKey.SerializeToString,
      ),
      'SubmitNewLivePolicyForPayoffTable': grpc.unary_unary_rpc_method_handler(
          servicer.SubmitNewLivePolicyForPayoffTable,
          request_deserializer=real__time__payoff__tracker__pb2.LivePolicyInfo.FromString,
          response_serializer=real__time__payoff__tracker__pb2.Confirmation.SerializeToString,
      ),
      'SubmitLiveEvalMatchupResult': grpc.unary_unary_rpc_method_handler(
          servicer.SubmitLiveEvalMatchupResult,
          request_deserializer=real__time__payoff__tracker__pb2.EvalMatchupResult.FromString,
          response_serializer=real__time__payoff__tracker__pb2.Confirmation.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'RealTimePayoffTracker', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
