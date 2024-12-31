import tensorflow as tf
import numpy as np
from QuantumChessGame import QuantumChessGame
from AZ_utils import to_observation

class AZConfig(object):

  def __init__(self):
    ### Self-Play
    self.num_actors = 5000

    self.num_sampling_moves = 30
    self.max_moves = 512  # for chess and shogi, 722 for Go.
    self.num_simulations = 800

    # Root prior exploration noise.
    self.root_dirichlet_alpha = 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
    self.root_exploration_fraction = 0.25

    # UCB formula
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    ### Training
    self.training_steps = int(700e3)
    self.checkpoint_interval = int(1e3)
    self.window_size = int(1e6)
    self.batch_size = 4096

    self.weight_decay = 1e-4
    self.momentum = 0.9
    # Schedule for chess and shogi, Go starts at 2e-2 immediately.
    self.learning_rate_schedule = {
        0: 2e-1,
        100e3: 2e-2,
        300e3: 2e-3,
        500e3: 2e-4
    }


class NN(object):
    def __init__(self, config, num_res_blocks=19) -> None:
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate_schedule[0], momentum=config.momentum)
        self.iterations = 0
        self.res_blocks = num_res_blocks
        self.network = self.make_network()
        self.config = config

    def make_network(self):
        # 6 piece types * 2 colors + 2 repetitions + 2 castling + 1 color + 1 move count + 1 no progress count + 128 entanglement + 1 probability
        number_states = int(145) # Aligns with the AZ utils currently
        input_shape = (8, 8, number_states)
        ins = tf.keras.layers.Input(shape=input_shape)
        hidden = tf.keras.layers.Conv2D(256, (3, 3), strides=1, padding="same")(ins)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.ReLU()(hidden)
        for block in range(self.res_blocks):
            block = tf.keras.layers.Conv2D(256, (3, 3), strides=1, padding="same")(hidden)
            block = tf.keras.layers.BatchNormalization()(block)
            block = tf.keras.layers.ReLU()(block)
            block = tf.keras.layers.Conv2D(256, (3, 3), strides=1, padding="same")(block)
            block = tf.keras.layers.BatchNormalization()(block)
            hidden = block + hidden
            hidden = tf.keras.layers.ReLU()(hidden)

        value_head = tf.keras.layers.Conv2D(1, (1, 1), strides=1, padding="same")(hidden)
        value_head = tf.keras.layers.BatchNormalization()(value_head)
        value_head = tf.keras.layers.ReLU()(value_head)
        value_head = tf.keras.layers.Flatten()(value_head)
        value_head = tf.keras.layers.Dense(256, activation='relu')(value_head)
        value_head = tf.keras.layers.Dense(1, activation='tanh')(value_head)

        policy_head = tf.keras.layers.Conv2D(8, (1, 1), strides=1, padding="same")(hidden)
        policy_head = tf.keras.layers.BatchNormalization()(policy_head)
        policy_head = tf.keras.layers.ReLU()(policy_head)
        policy_head = tf.keras.layers.Flatten()(policy_head)
        policy_head = tf.keras.layers.Dense(23360)(policy_head)

        model = tf.keras.models.Model(inputs=ins, outputs=[policy_head, value_head])
        #model.summary()
        return model

    def inference(self, image):
        return self.network(image)

    #@tf.function
    def update_weights(self, image, target_policy, target_value, weight_decay):
        loss = 0
        with tf.GradientTape() as tape:
            policy_logits, value = self.network(image)
            loss += (
                tf.math.reduce_mean(tf.math.square(value - target_value)) +
                tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=policy_logits, labels=target_policy))
            )

            for weights in self.network.trainable_variables:
                loss += weight_decay * tf.nn.l2_loss(weights)
        grads = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))

        return loss

    def train(self, replay_buffer):
        checkpoint_ = './checkpoints/'
        k = self.config.learning_rate_schedule.keys()
        for i in range(self.config.training_steps):
            if i > k[1]:
                self.optimizer.lr = self.config.learning_rate_schedule[k[1]]
            if i > k[2]:
                self.optimizer.lr = self.config.learning_rate_schedule[k[2]]
            if i > k[3]:
                self.optimizer.lr = self.config.learning_rate_schedule[k[3]]
            if i % self.config.checkpoint_interval == 0:
                checkpoint_path = checkpoint_ + str(i)
                self.network.save_weights(checkpoint_path)
                im, tp1, tp2, v = replay_buffer.sample_batch()
                self.update_weights(im, tp1, tp2, v, self.config.weight_decay)
        self.network.save_weights(checkpoint_)

if __name__ == "__main__":
    import random
    test = NN(AZConfig())
    '''
    x = test.inference(np.random.uniform(-1, 1, (10, 8, 8, 81)))
    print([i.shape for i in x])
    #test.network.summary()
    tp1 = np.zeros(shape=(10, 4672))
    for i in range(10):
        tp1[i][np.random.randint(0, 4672)] = 1
    im, v = [np.random.uniform(-1, 1, (10, 8, 8, 81)), np.random.uniform(-1, 1, (10, 1))]
    for i in range(10):
        l = test.update_weights(im, tp1, v, 1e-4)
        print("LOSS", l)
    '''
    replay_buffer = np.zeros(shape=(10, 8, 8, 145))
    counter = 0
    game = QuantumChessGame()
    game.new_game()

    while True:
        valid_moves = game.get_legal_moves()
        move = random.choice(valid_moves)
        gamedata, move_code = game.do_move(move)

        game.print_probability_board()
        reprs = to_observation(game)
        replay_buffer[counter % 10] = reprs
        counter += 1

        if counter > 20:
            break
        #print(counter)
        #print(reprs.shape)
        #print(test.inference(np.expand_dims(reprs, axis=0)))

    #for i in range(10):
    #    l = test.update_weights(replay_buffer, tp1, 0, 1e-4)
    #    print("LOSS", l)
