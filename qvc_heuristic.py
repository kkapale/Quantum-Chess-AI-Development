import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
import cirq
import sympy
import matplotlib.pyplot as plt
from heuristics import *


class ReUp(tf.keras.layers.Layer):
    def __init__(self, num_qubits, depth, input_size=64) -> None:
        super(ReUp, self).__init__()
        self.layers = depth
        self.num_qubits = num_qubits
        self.inputs = input_size
        self.layer_params = 3 * self.layers * self.num_qubits
        self.lambd = tf.Variable(initial_value=np.random.uniform(0, 2 * np.pi, (1, self.inputs, self.layer_params)), dtype="float32", trainable=True)
        self.theta = tf.Variable(initial_value=np.random.uniform(0, 2 * np.pi, (1, self.layer_params)), dtype="float32", trainable=True)
        self.w = tf.Variable(initial_value=np.random.uniform(-1, 1, (self.num_qubits, 1)), dtype="float32", trainable=True)
        self.g = 2
        self.num_params = 3 * self.g * self.layers * self.num_qubits
        self.qubits = [cirq.GridQubit(0, i) for i in range(self.num_qubits)]
        self.params = sympy.symbols("params0:%d"%self.num_params)
        self.model = tfq.layers.ControlledPQC(self.make_circuit(self.params), [cirq.Z(i) for i in self.qubits], differentiator=tfq.differentiators.Adjoint())
        self.in_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.indices = []
        i = 0
        while i < self.layer_params:
            for j in range(self.num_qubits * 3):
                self.indices.append(i + j)
            for j in range(self.num_qubits * 3):
                self.indices.append(i + j + self.layer_params)
            i += self.num_qubits * 3

    def make_circuit(self, params):
        cir = cirq.Circuit()
        for i in self.qubits:
            cir += cirq.H(i)
        params_per_layer = 3 * 2 * self.num_qubits
        p = 0
        for i in range(self.layers):
            cir += self.u_ent(params[p:p + params_per_layer//2])
            cir += self.u_enc(params[p + params_per_layer//2:p + params_per_layer])
            p += params_per_layer
            if i == 0:
                print(cir)
        return cir

    def u_ent(self, ps):
        c = cirq.Circuit()
        for i in range(self.num_qubits):
            c += cirq.rz(ps[i]).on(self.qubits[i])
        for i in range(self.num_qubits):
            c += cirq.ry(ps[i + self.num_qubits]).on(self.qubits[i])
        for i in range(self.num_qubits):
            c += cirq.rz(ps[i + 2 * self.num_qubits]).on(self.qubits[i])
        for i in range(self.num_qubits - 1):
            c += cirq.CZ(self.qubits[i], self.qubits[i+1])
        c += cirq.CZ(self.qubits[-1], self.qubits[0])
        return c

    def u_enc(self, ps):
        c = cirq.Circuit()
        for i in range(self.num_qubits):
            c += cirq.ry(ps[i]).on(self.qubits[i])
        for i in range(self.num_qubits):
            c += cirq.rz(ps[i + self.num_qubits]).on(self.qubits[i])
        for i in range(self.num_qubits):
            c += cirq.rz(ps[i + 2 * self.num_qubits]).on(self.qubits[i])
        return c

    # inputs = (batch, in_size)
    def call(self, inputs):
        num_batch = tf.gather(tf.shape(inputs), 0)
        # (1, 1) -> (batch, 1)
        input_circuits = tf.repeat(self.in_circuit, repeats=num_batch)
        # (batch, in_size) -> (batch, 1, in_size)
        inputs = tf.expand_dims(inputs, axis=1)
        # (batch, 1, in_size) * (1, in_size, num_param) -> (batch, 1, num_params)
        lambs = tf.matmul(inputs, self.lambd)
        # (batch, 1, num_params) -> (batch, num_params)
        lambs = tf.squeeze(lambs, axis=1)
        # (1, num_param) -> (batch, num_params)
        thetas = tf.tile(self.theta, [num_batch, 1])
        # (batch, num_params), (batch, num_params) -> (batch, total_params)
        full_params = tf.concat([thetas, lambs], axis=1)
        full_params = tf.gather(full_params, self.indices, axis=1)
        # -> (batch, n_qubit)
        output = self.model([input_circuits, full_params])
        # (batch, n_qubit) -> (batch, 1)
        out = tf.linalg.matmul(output, self.w)
        return out

def train_material(pieces, probabilities, player):
    evaluation = sum([piece_values[pieces[i]] * probabilities[i] for i in range(len(pieces)) if pieces[i] != '.'])
    perspective = 1 if (player == 0) else -1
    return evaluation * perspective

def train_material_and_position(pieces, probabilities, player):
    material_score = train_material(pieces, probabilities, player)
    position_value = 0
    for i, p in enumerate(pieces):
        if p == '.':
            continue
        val = get_piece_position_value(p, i, True)
        position_value += val * probabilities[i]

    perspective = 1 if (player == 0) else -1
    return position_value * perspective + material_score

data_size = 1000
val = [".", "P", "p", "N", "n", "Q", "q", "R", "r", "B", "b", "K", "k"]
w = np.array([32, 8, 8, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1], dtype=np.float32)
w = (w * 1/np.sum(w))
data = np.random.choice(len(val), size=(data_size, 64), p=w)
vals = [[val[j] for j in i] for i in data]
#probs = np.random.uniform(0, 1, (data_size, 64))
probs = np.ones((data_size, 64))
y = np.array([train_material_and_position(vals[i], probs[i], 0) for i in range(data_size)])

reup = ReUp(3, 5)
inputs = tf.keras.Input(shape=(64,))
layer1 = reup(inputs)
vqc = tf.keras.models.Model(inputs=inputs, outputs=layer1)
vqc.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.01))
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


X_train = data[:9 * len(data) //10]
y_train = y[:9 * len(y)//10]
X_test = data[9 * len(data)//10:]
y_test = y[9 * len(y)//10:]

v_history = vqc.fit(X_train, tf.convert_to_tensor(y_train)/np.max(y_train), epochs=100, batch_size=32, validation_data=(X_test, tf.convert_to_tensor(y_test)/np.max(y_train)), \
    callbacks=[callback])


plt.plot(v_history.history['loss'], label='Quantum Training Loss')
plt.plot(v_history.history['val_loss'], label='Quantum Validation Loss')
plt.legend()
plt.show()
plt.savefig("boston_housing")
