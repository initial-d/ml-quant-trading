import torch
import torch.nn as nn
import torch.nn.functional as F
import codecs
import numpy as np

class LinearLayer(nn.Module) :
    def __init__(self, input_size, out_size, activate_func = None):
        super(LinearLayer, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.W  = nn.Linear(input_size, out_size)

        #nn.init.xavier_uniform_(self.W.weight)  # glorot
        #nn.init.zeros_(self.W.bias)

        self.activate_func = activate_func       
    def forward(self, intput_tensor):
        if self.activate_func is None:
            return self.W(intput_tensor)
        return self.activate_func(self.W(intput_tensor))


class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.
    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.
    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:
            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`
    Example:
         # >>> attention = Attention(256)
         # >>> query = torch.randn(5, 1, 256)
         # >>> context = torch.randn(5, 5, 256)
         # >>> output, weights = attention(query, context)
         # >>> output.size()
         # torch.Size([5, 1, 256])
         # >>> weights.size()
         # torch.Size([5, 1, 5])
    """
    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()
        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')
        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)
        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.
        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)
        if self.attention_type == "general":
            query = query.view(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.view(batch_size, output_len, dimensions)
        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())
        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)
        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)
        return mix


class StockPred_v1(nn.Module) :
    def __init__(self):
        super(StockPred_v1, self).__init__()
        self.layer_1 = LinearLayer(96, 64, F.relu)
        self.layer_2 = LinearLayer(64, 32, F.relu)
        self.layer_3 = LinearLayer(32, 8, F.relu)
        self.layer_4 = LinearLayer(8, 1, None)

    def forward(self, data):
        d1_out = self.layer_1(data)
        d2_out = self.layer_2(d1_out)
        d3_out = self.layer_3(d2_out)
        d4_out = self.layer_4(d3_out)
        return d4_out

    def valid_batch(self, data):
        x = self.forward(data)
        return F.sigmoid(x)

    def predict(self, data):
        d1_out = self.layer_1(data)
        d2_out = self.layer_2(d1_out)
        d3_out = self.layer_3(d2_out)
        d4_out = self.layer_4(d3_out)
        dnn_output = F.sigmoid(d4_out)
        return dnn_output


class StockPred_trend(nn.Module):
    def __init__(self):
        super(StockPred_trend, self).__init__()
        self.layer_1 = LinearLayer(96 + 1, 64, F.relu)
        self.layer_2 = LinearLayer(64, 32, F.relu)
        self.layer_3 = LinearLayer(32, 8, F.relu)
        self.layer_4 = LinearLayer(8, 1, None)

    def forward(self, data):
        d1_out = self.layer_1(data)
        d2_out = self.layer_2(d1_out)
        d3_out = self.layer_3(d2_out)
        d4_out = self.layer_4(d3_out)
        return d4_out

    def valid_batch(self, data):
        x = self.forward(data)
        return F.sigmoid(x)

    def predict(self, data):
        d1_out = self.layer_1(data)
        d2_out = self.layer_2(d1_out)
        d3_out = self.layer_3(d2_out)
        d4_out = self.layer_4(d3_out)
        dnn_output = F.sigmoid(d4_out)
        return dnn_output



class StockPred_v2(nn.Module) :
    def __init__(self):
        super(StockPred_v2, self).__init__()
        # self.quantity_emb = nn.Embedding(100, 128)
        # self.cnt_emb = nn.Embedding(100, 128)
        # self.val_emb = nn.Embedding(100, 128)
        self.layer_1 = LinearLayer(9186, 1024, F.relu)
        self.layer_2 = LinearLayer(1024, 512, F.relu)
        self.layer_3 = LinearLayer(512, 128, F.relu)
        self.layer_4 = LinearLayer(128, 32, F.relu)
        self.layer_5 = LinearLayer(32, 1, None)

    def forward(self, id_data, data):
        all_data = torch.cat([id_data, data], 1)
        d1_out = self.layer_1(all_data)
        d2_out = self.layer_2(d1_out)
        d3_out = self.layer_3(d2_out)
        d4_out = self.layer_4(d3_out)
        d5_out = self.layer_5(d4_out)
        return d5_out

    def valid_batch(self, id_data, data):
        x = self.forward(id_data, data)
        return F.sigmoid(x)

    def predict(self, id_data, data):
        all_data = torch.cat([id_data, data], 1)
        d1_out = self.layer_1(all_data)
        d2_out = self.layer_2(d1_out)
        d3_out = self.layer_3(d2_out)
        d4_out = self.layer_4(d3_out)
        d5_out = self.layer_5(d4_out)
        dnn_output = F.sigmoid(d5_out)
        return dnn_output


class StockPred_v3(nn.Module):
    def __init__(self):
        super(StockPred_v3, self).__init__()
        
        #IN = codecs.open('model', 'r', encoding='utf-8')  #utf-8

        #line = IN.readline()
        #wl = []
        #while line:
        #    line = line.rstrip()
        #    #hs300_list.append(line) 
        #    wl.append(list(map(lambda x:float(x), line.split(' '))))
        #    line = IN.readline()
        #IN.close()

        #wnpl = []
        #for l in wl:
        #    npl = np.array(l)
        #    norm = np.linalg.norm(npl)
        #    normal_npl = npl / norm
        #    wnpl.append(normal_npl)
        #wl = wnpl
        
        
        self.layer_1 = LinearLayer(144, 256, torch.sigmoid)
        self.layer_2 = LinearLayer(256, 256, torch.sigmoid)
        self.layer_3 = LinearLayer(256, 256, torch.sigmoid)
        self.layer_4 = LinearLayer(256, 256, torch.sigmoid)
        self.layer_5 = LinearLayer(256, 128, torch.sigmoid)
        self.layer_6 = LinearLayer(128, 64, torch.sigmoid)
        self.layer_7 = LinearLayer(64, 1, None)
       
        #self.layer_1.W.weight = torch.nn.Parameter(torch.from_numpy(np.reshape(wl[0], (512, 159))))
        #self.layer_1.W.bias = torch.nn.Parameter(torch.from_numpy(np.array(wl[1])))
        ###print(self.layer_1.W.bias.tolist())
        ##print("weight############")
        ##print(self.layer_1.W.weight.tolist())
        ##print("bias############")
        ##print(self.layer_1.W.bias.tolist())

        #self.layer_2.W.weight = torch.nn.Parameter(torch.from_numpy(np.reshape(wl[2], (256, 512))))
        #self.layer_2.W.bias = torch.nn.Parameter(torch.from_numpy(np.array(wl[3])))

        #self.layer_3.W.weight = torch.nn.Parameter(torch.from_numpy(np.reshape(wl[4], (128, 256))))
        #self.layer_3.W.bias = torch.nn.Parameter(torch.from_numpy(np.array(wl[5])))

        #self.layer_4.W.weight = torch.nn.Parameter(torch.from_numpy(np.reshape(wl[6], (64, 128))))
        #self.layer_4.W.bias = torch.nn.Parameter(torch.from_numpy(np.array(wl[7])))

        #self.layer_5.W.weight = torch.nn.Parameter(torch.from_numpy(np.reshape(wl[8], (1, 64))))
        #self.layer_5.W.bias = torch.nn.Parameter(torch.from_numpy(np.array(wl[9])))
        ##self.layer_5.W.bias = torch.nn.Parameter(torch.from_numpy(np.array(0.0)))


        #self.layer_1.W.weight = torch.nn.Parameter(torch.from_numpy(np.transpose(np.reshape(wl[0], (159, 512)))))
        #self.layer_1.W.bias = torch.nn.Parameter(torch.from_numpy(np.array(wl[1])))
        ##print("weight############")
        ##print(self.layer_1.W.weight.tolist())
        ##print("bias############")
        ##print(self.layer_1.W.bias.tolist())

        #self.layer_2.W.weight = torch.nn.Parameter(torch.from_numpy(np.transpose(np.reshape(wl[2], (512, 256)))))
        #self.layer_2.W.bias = torch.nn.Parameter(torch.from_numpy(np.array(wl[3])))

        #self.layer_3.W.weight = torch.nn.Parameter(torch.from_numpy(np.transpose(np.reshape(wl[4], (256, 128)))))
        #self.layer_3.W.bias = torch.nn.Parameter(torch.from_numpy(np.array(wl[5])))

        #self.layer_4.W.weight = torch.nn.Parameter(torch.from_numpy(np.transpose(np.reshape(wl[6], (128, 64)))))
        #self.layer_4.W.bias = torch.nn.Parameter(torch.from_numpy(np.array(wl[7])))

        #self.layer_5.W.weight = torch.nn.Parameter(torch.from_numpy(np.transpose(np.reshape(wl[8], (64, 1)))))
        #self.layer_5.W.bias = torch.nn.Parameter(torch.from_numpy(np.array(wl[9])))


    def forward(self, data):
        d1_out = self.layer_1(data)
        #print("out###########")
        #print(len(d1_out.permute(1, 0).tolist()))
        #print(d1_out.permute(1, 0).tolist())
        d2_out = self.layer_2(d1_out)
        d3_out = self.layer_3(d2_out)
        d4_out = self.layer_4(d3_out)
        d5_out = self.layer_5(d4_out)
        d6_out = self.layer_6(d5_out)
        d7_out = self.layer_7(d6_out)
        return d7_out

    def valid_batch(self, data):
        x = self.forward(data)
        return x

    def predict(self, data):
        d1_out = self.layer_1(data)
        d2_out = self.layer_2(d1_out)
        d3_out = self.layer_3(d2_out)
        d4_out = self.layer_4(d3_out)
        d5_out = self.layer_5(d4_out)
        d6_out = self.layer_6(d5_out)
        d7_out = self.layer_7(d6_out)
        dnn_output = d7_out
        return dnn_output
