import torch
import torch.nn as nn
from torch.autograd import Variable
from data_loading import *
from torch.utils.tensorboard import SummaryWriter 
import os
'''
the model
'''
class generative_model(nn.Module):
	def __init__(self, vocabs_size, hidden_size, output_size, embedding_dimension, n_layers):
		super(generative_model, self).__init__()
		self.vocabs_size = vocabs_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.embedding_dimension = embedding_dimension
		self.n_layers = n_layers

		self.embedding = nn.Embedding(vocabs_size, embedding_dimension)
		self.rnn = nn.LSTM(embedding_dimension, hidden_size, n_layers, dropout = 0.2)
		self.linear = nn.Linear(hidden_size, output_size)
	
	def forward(self, input, hidden):
		batch_size = input.size(0)
		input = self.embedding(input)
		output, hidden = self.rnn(input.view(1, batch_size, -1), hidden)
		output = self.linear(output.view(batch_size, -1))
		return output, hidden

	def init_hidden(self, batch_size):
		hidden=(Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
				Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
		return hidden

data,vocabs=load_data()
data=data[-1000:]  ### smaller dataset  
vocabs = list(vocabs)
vocabs_size = len(vocabs)
output_size = len(vocabs)
batch_size = 128
#batch_size = 20
cuda = False
hidden_size = 1024
embedding_dimension =  248
n_layers=3
#lr = 0.0001 
lr = 0.0001
#n_batches=5
n_batches =  200000
#print_every = 100
print_every = 100
plot_every = 100
#save_every = 100
save_every = 1000#1000
end_token = ' '

print("processing batches ...")
train_batches,val_batches = process_data_to_batches(data,batch_size,vocabs,cuda)
print("finish processing batches")


model = generative_model(vocabs_size,hidden_size,output_size,embedding_dimension,n_layers)
criterion = nn.CrossEntropyLoss()

if cuda:
	model = model.cuda()
	criterion = criterion.cuda()




def propagation(inp, target, mode):
	batch_size = inp.size(0)
	sequence_length = inp.size(1)
	hidden = model.init_hidden(batch_size)
	if cuda:
		hidden = (hidden[0].cuda(), hidden[1].cuda())
	if mode=='train':
		model.zero_grad()
	loss = 0
	for c in range(sequence_length):
		output, hidden = model(inp[:,c], hidden)
		loss += criterion(output, target[:,c])
	return loss.item()/sequence_length
#    return loss.data[0]/sequence_length


def evaluate(prime_str='!', temperature=0.4):
	max_length = 200
	if cuda:
		inp = Variable(tensor_from_chars_list(prime_str,vocabs,cuda)).cuda()
	else:
		inp = Variable(tensor_from_chars_list(prime_str,vocabs,cuda))
	batch_size = inp.size(0)
	hidden = model.init_hidden(batch_size)
	if cuda:
		hidden = (hidden[0].cuda(), hidden[1].cuda())
	predicted = prime_str
	while True:
		output, hidden = model(inp, hidden)
		# Sample from the network as a multinomial distribution
		output_dist = output.data.view(-1).div(temperature).exp()
		top_i = torch.multinomial(output_dist, 1)[0]
		# Add predicted character to string and use as next input
		predicted_char = vocabs[top_i]

		if predicted_char ==end_token or len(predicted)>max_length:
			return predicted

		predicted += predicted_char
		if cuda:
			inp = Variable(tensor_from_chars_list(predicted_char,vocabs,cuda)).cuda()
		else:
			inp = Variable(tensor_from_chars_list(predicted_char,vocabs,cuda))
	return predicted

start = time.time()




with torch.no_grad():
	model_files=os.listdir('./Pretraining2/models')
	model_dir='./Pretraining2/models/'
	for model_path in model_files:
		n_batch=int(model_path.split('.')[0].split('_')[1])
		model.load_state_dict(torch.load(model_dir+model_path, map_location=torch.device('cpu')))
		inp, target = get_random_batch(val_batches)
		loss = propagation(inp, target, 'train')  
		print(n_batch,loss)   
		if n_batch % 5000 == 0:
			print(evaluate('!'), '\n')
			print(n_batch,loss)
