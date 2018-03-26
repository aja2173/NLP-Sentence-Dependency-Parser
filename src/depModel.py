import os,sys
from decoder import *

class DepModel:
    def __init__(self):
        '''
            You can add more arguments for examples actions and model paths.
            You need to load your model here.
            actions: provides indices for actions.
            it has the same order as the data/vocabs.actions file.
        '''
        # if you prefer to have your own index for actions, change this.
        self.actions = ['SHIFT', 'LEFT-ARC:rroot', 'LEFT-ARC:cc', 'LEFT-ARC:number', 'LEFT-ARC:ccomp', 'LEFT-ARC:possessive', 'LEFT-ARC:prt', 'LEFT-ARC:num', 'LEFT-ARC:nsubjpass', 'LEFT-ARC:csubj', 'LEFT-ARC:conj', 'LEFT-ARC:dobj', 'LEFT-ARC:nn', 'LEFT-ARC:neg', 'LEFT-ARC:discourse', 'LEFT-ARC:mark', 'LEFT-ARC:auxpass', 'LEFT-ARC:infmod', 'LEFT-ARC:mwe', 'LEFT-ARC:advcl', 'LEFT-ARC:aux', 'LEFT-ARC:prep', 'LEFT-ARC:parataxis', 'LEFT-ARC:nsubj', 'LEFT-ARC:<null>', 'LEFT-ARC:rcmod', 'LEFT-ARC:advmod', 'LEFT-ARC:punct', 'LEFT-ARC:quantmod', 'LEFT-ARC:tmod', 'LEFT-ARC:acomp', 'LEFT-ARC:pcomp', 'LEFT-ARC:poss', 'LEFT-ARC:npadvmod', 'LEFT-ARC:xcomp', 'LEFT-ARC:cop', 'LEFT-ARC:partmod', 'LEFT-ARC:dep', 'LEFT-ARC:appos', 'LEFT-ARC:det', 'LEFT-ARC:amod', 'LEFT-ARC:pobj', 'LEFT-ARC:iobj', 'LEFT-ARC:expl', 'LEFT-ARC:predet', 'LEFT-ARC:preconj', 'LEFT-ARC:root', 'RIGHT-ARC:rroot', 'RIGHT-ARC:cc', 'RIGHT-ARC:number', 'RIGHT-ARC:ccomp', 'RIGHT-ARC:possessive', 'RIGHT-ARC:prt', 'RIGHT-ARC:num', 'RIGHT-ARC:nsubjpass', 'RIGHT-ARC:csubj', 'RIGHT-ARC:conj', 'RIGHT-ARC:dobj', 'RIGHT-ARC:nn', 'RIGHT-ARC:neg', 'RIGHT-ARC:discourse', 'RIGHT-ARC:mark', 'RIGHT-ARC:auxpass', 'RIGHT-ARC:infmod', 'RIGHT-ARC:mwe', 'RIGHT-ARC:advcl', 'RIGHT-ARC:aux', 'RIGHT-ARC:prep', 'RIGHT-ARC:parataxis', 'RIGHT-ARC:nsubj', 'RIGHT-ARC:<null>', 'RIGHT-ARC:rcmod', 'RIGHT-ARC:advmod', 'RIGHT-ARC:punct', 'RIGHT-ARC:quantmod', 'RIGHT-ARC:tmod', 'RIGHT-ARC:acomp', 'RIGHT-ARC:pcomp', 'RIGHT-ARC:poss', 'RIGHT-ARC:npadvmod', 'RIGHT-ARC:xcomp', 'RIGHT-ARC:cop', 'RIGHT-ARC:partmod', 'RIGHT-ARC:dep', 'RIGHT-ARC:appos', 'RIGHT-ARC:det', 'RIGHT-ARC:amod', 'RIGHT-ARC:pobj', 'RIGHT-ARC:iobj', 'RIGHT-ARC:expl', 'RIGHT-ARC:predet', 'RIGHT-ARC:preconj', 'RIGHT-ARC:root']
        # write your code here for additional parameters.
        # feel free to add more arguments to the initializer.
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

        self.words = words
        self.pos=pos
        self.labels=labels

        import dynet as dynet
        import random

        self.model = dynet.Model()

        num_words = 4807
        num_pos = 45
        num_labels = 46
        num_actions = 93
        word_embed_dim = 64
        pos_embed_dim = 32
        label_embed_dim = 32

        self.word_embedding = self.model.add_lookup_parameters((num_words, word_embed_dim))
        self.pos_embedding = self.model.add_lookup_parameters((num_pos, pos_embed_dim))
        self.label_embedding = self.model.add_lookup_parameters((num_labels, label_embed_dim))

        self.transfer = dynet.rectify

        input_dim = 20*(word_embed_dim+pos_embed_dim)+12*label_embed_dim
        hidden_dim1 = 600
        hidden_dim2 = 600
        minibatch_size = 1000

        self.hidden_layer1 = self.model.add_parameters((hidden_dim1, input_dim))
        self.hidden_layer1_bias = self.model.add_parameters(hidden_dim1, init=dynet.ConstInitializer(0.2))
        self.hidden_layer2 = self.model.add_parameters((hidden_dim2, hidden_dim1))
        self.hidden_layer2_bias = self.model.add_parameters(hidden_dim2, init=dynet.ConstInitializer(0.2))

        self.output_layer = self.model.add_parameters((num_actions, hidden_dim2))
        self.output_bias = self.model.add_parameters(num_actions,init=dynet.ConstInitializer(0))
        self.model.populate('./NNpart3')

    def score(self, features):
        '''
        :param str_features: String features
        20 first: words, next 20: pos, next 12: dependency labels.
        DO NOT ADD ANY ARGUMENTS TO THIS FUNCTION.
        :return: list of scores
        '''
        # change this part of the code.
        import dynet as dynet
        transfer = dynet.rectify
        words = self.words
        pos=self.pos
        labels=self.labels
        actions=self.actions
        def word2id(word):
            return words[word] if word in words else words['<unk>']
        def pos2id(tag):
            return pos[tag] if tag in pos else pos['<null>']
        def label2id(label):
            return labels[label] if label in labels else labels['<null>']
        def action2id(action):
            return actions[action]

        word_ids = [word2id(word_feat) for word_feat in features[0:20]]
        pos_ids = [pos2id(pos_feat) for pos_feat in features[20:40]]
        label_ids = [label2id(label_feat) for label_feat in features[40:52]]

        word_embeds = [self.word_embedding[wid] for wid in word_ids]
        pos_embeds = [self.pos_embedding[pid] for pid in pos_ids]
        label_embeds = [self.label_embedding[lid] for lid in label_ids]

        embedding_layer = dynet.concatenate(word_embeds+pos_embeds+label_embeds)

        hidden1 = transfer(self.hidden_layer1.expr()*embedding_layer + self.hidden_layer1_bias.expr())
        hidden2 = transfer(self.hidden_layer2.expr()*hidden1 + self.hidden_layer2_bias.expr())

        result = self.output_layer.expr()*hidden2 + self.output_bias.expr()

        output = result.npvalue()
        #print(np.argmax(output))
        return output

if __name__=='__main__':
    m = DepModel()
    input_p = os.path.abspath(sys.argv[1])
    output_p = os.path.abspath(sys.argv[2])
    Decoder(m.score, m.actions).parse(input_p, output_p)