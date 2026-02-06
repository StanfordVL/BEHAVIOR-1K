import zmq
import pickle

# Create a ZMQ context
context = zmq.Context()

# Create a socket to receive messages (SUB)
data_queue = context.socket(zmq.SUB)
data_queue.connect("tcp://127.0.0.1:5555")

# Subscribe to all messages
data_queue.setsockopt_string(zmq.SUBSCRIBE, "")

# Create a socket to send messages (PUB)
state_queue = context.socket(zmq.PUB)
state_queue.bind("tcp://127.0.0.1:5556")

with open("data.pickle", "rb") as file:
    state = pickle.load(file)

while True:
    state_queue.send(pickle.dumps(state))
    # Receive a message
    data = data_queue.recv()
    data = pickle.loads(data)
    # Prints rewards
    print([d[2] for d in data])
