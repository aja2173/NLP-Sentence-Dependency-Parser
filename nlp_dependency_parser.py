#NLP-Dependency-Parser
import numpy as np

#Creating Dictionaries
word_path = './data/vocabs.word'
pos_path = './data/vocabs.pos'
labels_path = './data/vocabs.labels'
actions_path = './data/vocabs.actions'

words = {}
pos = {}
labels = {}
actions = {}

with open(word_path,'r') as file:
	for line in file:
		line = line[0:len(line)-1]
		line = line.split(' ')
		index = int(line[1])
		word = line[0]
		words[word]=index

with open(pos_path,'r') as file:
	for line in file:
		line = line[0:len(line)-1]
		line = line.split(' ')
		index = int(line[1])
		word = line[0]
		pos[word]=index

with open(labels_path,'r') as file:
	for line in file:
		line = line[0:len(line)-1]
		line = line.split(' ')
		index = int(line[1])
		word = line[0]
		labels[word]=index

with open(actions_path,'r') as file:
	for line in file:
		line = line[0:len(line)-1]
		line = line.split(' ')
		index = int(line[1])
		word = line[0]
		actions[word]=index

print('Dictionaries Created')

#Creating auxililary functions

def word2id(word):
	return words[word] if word in words else words['<unk>']
def pos2id(tag):
	return pos[tag]
def label2id(label):
	return labels[label]
def action2id(action):
	return actions[action]

num_words = 4807
num_pos = 45
num_labels = 46
num_actions = 93

#Dynet Network
print('Starting dynet part')
import dynet as dynet
import random

model = dynet.Model()
updater = dynet.AdamTrainer(model)

word_embed_dim = 64
pos_embed_dim = 32
label_embed_dim = 32

word_embedding = model.add_lookup_parameters((num_words, word_embed_dim))
pos_embedding = model.add_lookup_parameters((num_pos, pos_embed_dim))
label_embedding = model.add_lookup_parameters((num_labels, label_embed_dim))

transfer = dynet.rectify

input_dim = 20*(word_embed_dim+pos_embed_dim)+12*label_embed_dim
hidden_dim1 = 400
hidden_dim2 = 400
minibatch_size = 1000

hidden_layer1 = model.add_parameters((hidden_dim1, input_dim))
hidden_layer1_bias = model.add_parameters(hidden_dim1, init=dynet.ConstInitializer(0.2))
hidden_layer2 = model.add_parameters((hidden_dim2, hidden_dim1))
hidden_layer2_bias = model.add_parameters(hidden_dim2, init=dynet.ConstInitializer(0.2))

output_layer = model.add_parameters((num_actions, hidden_dim2))
output_bias = model.add_parameters(num_actions,init=dynet.ConstInitializer(0))

#Forward Function
def forward(features):
	word_ids = [word2id(word_feat) for word_feat in features[0:20]]
	pos_ids = [pos2id(pos_feat) for pos_feat in features[20:40]]
	label_ids = [label2id(label_feat) for label_feat in features[40:52]]

	word_embeds = [word_embedding[wid] for wid in word_ids]
	pos_embeds = [pos_embedding[pid] for pid in pos_ids]
	label_embeds = [label_embedding[lid] for lid in label_ids]

	embedding_layer = dynet.concatenate(word_embeds+pos_embeds+label_embeds)

	hidden1 = transfer(hidden_layer1.expr()*embedding_layer + hidden_layer1_bias.expr())
	hidden2 = transfer(hidden_layer2.expr()*hidden1 + hidden_layer2_bias.expr())

	output = output_layer.expr()*hidden2 + output_bias.expr()

	return output

#Training the model
train_data_path = './data/train.data'
train_data = open(train_data_path,'r').read().strip().split('\n')
def train_iter(train_data):
	losses = []
	random.shuffle(train_data)

	for line in train_data:
		fields = line.strip().split(' ')
		features, gold_label = fields[:-1], action2id(fields[-1])

		result = forward(features)
		#print(result.value())
		loss = dynet.pickneglogsoftmax(result, gold_label)
		losses.append(loss)

		if len(losses) >= minibatch_size:
			minibatch_loss = dynet.esum(losses) / len(losses)
			minibatch_loss.forward()
			minibatch_loss_value = minibatch_loss.value()
			print(minibatch_loss_value)
			minibatch_loss.backward()
			updater.update()

			losses = []
			dynet.renew_cg()
	dynet.renew_cg()

num_epochs = 7

for i in range(num_epochs):
	print('epoch',i+1)
	train_iter(train_data)
	dynet.renew_cg()

print('finished training!')

def load(filename):
	model.populate(filename)

def save(filename):
	model.save(filename)

outfile = 'NNpart2'

save(outfile)


