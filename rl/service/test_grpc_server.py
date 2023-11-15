from concurrent import futures
import logging

import grpc
import omnigibson_pb2
import omnigibson_pb2_grpc

from PIL import Image
import io

class Policy(omnigibson_pb2_grpc.PolicyServicer):
  def Step(self, request, context):
    # Get the image
    img = Image.open(io.BytesIO(request.image))
    resp = omnigibson_pb2.StepResponse()
    print("Got req")
    resp.command.extend([0] * 22)
    return resp


def serve():
  port = "50051"
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
  omnigibson_pb2_grpc.add_PolicyServicer_to_server(Policy(), server)
  server.add_insecure_port("[::]:" + port)
  server.start()
  print("Server started, listening on " + port)
  server.wait_for_termination()


if __name__ == "__main__":
  logging.basicConfig()
  serve()