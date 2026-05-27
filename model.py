import torch
import torch.nn as nn
from transformers import BertModel

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(CustomLSTM, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout

        self.W_f = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size, input_size if i == 0 else hidden_size) * 0.01)
            for i in range(num_layers)
        ])
        self.U_f = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
            for _ in range(num_layers)
        ])
        self.b_f = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_size))
            for _ in range(num_layers)
        ])
        self.W_i = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size, input_size if i == 0 else hidden_size) * 0.01)
            for i in range(num_layers)
        ])
        self.U_i = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
            for _ in range(num_layers)
        ])
        self.b_i = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_size))
            for _ in range(num_layers)
        ])
        self.W_c = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size, input_size if i == 0 else hidden_size) * 0.01)
            for i in range(num_layers)
        ])
        self.U_c = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
            for _ in range(num_layers)
        ])
        self.b_c = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_size))
            for _ in range(num_layers)
        ])
        self.W_o = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size, input_size if i == 0 else hidden_size) * 0.01)
            for i in range(num_layers)
        ])
        self.U_o = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
            for _ in range(num_layers)
        ])
        self.b_o = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_size))
            for _ in range(num_layers)
        ])

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        self._init_weights()

    def _init_weights(self):
        for layer in range(self.num_layers):
            nn.init.xavier_uniform_(self.W_f[layer])
            nn.init.xavier_uniform_(self.U_f[layer])
            nn.init.xavier_uniform_(self.W_i[layer])
            nn.init.xavier_uniform_(self.U_i[layer])
            nn.init.xavier_uniform_(self.W_c[layer])
            nn.init.xavier_uniform_(self.U_c[layer])
            nn.init.xavier_uniform_(self.W_o[layer])
            nn.init.xavier_uniform_(self.U_o[layer])

    def forward_layer(self, x, h, c, layer_idx):
        batch_size, seq_len, _ = x.size()
        hidden_seq = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            f_t = torch.sigmoid(
                torch.matmul(x_t, self.W_f[layer_idx].t()) +
                torch.matmul(h,   self.U_f[layer_idx].t()) +
                self.b_f[layer_idx]
            )
            i_t = torch.sigmoid(
                torch.matmul(x_t, self.W_i[layer_idx].t()) +
                torch.matmul(h,   self.U_i[layer_idx].t()) +
                self.b_i[layer_idx]
            )
            c_tilde = torch.tanh(
                torch.matmul(x_t, self.W_c[layer_idx].t()) +
                torch.matmul(h,   self.U_c[layer_idx].t()) +
                self.b_c[layer_idx]
            )
            c   = f_t * c + i_t * c_tilde
            o_t = torch.sigmoid(
                torch.matmul(x_t, self.W_o[layer_idx].t()) +
                torch.matmul(h,   self.U_o[layer_idx].t()) +
                self.b_o[layer_idx]
            )
            h   = o_t * torch.tanh(c)
            hidden_seq.append(h.unsqueeze(1))
        hidden_seq = torch.cat(hidden_seq, dim=1)
        return hidden_seq, h, c

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()
        if hidden is None:
            h = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        else:
            h, c = hidden
        current_input = x
        for layer in range(self.num_layers):
            hidden_seq, h[layer], c[layer] = self.forward_layer(
                current_input, h[layer], c[layer], layer
            )
            if self.dropout_layer is not None and layer < self.num_layers - 1:
                hidden_seq = self.dropout_layer(hidden_seq)
            current_input = hidden_seq
        return hidden_seq, (h, c)


class BERTLSTMClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes, hidden_dim, num_layers, dropout=0.3, freeze_bert=True):
        super(BERTLSTMClassifier, self).__init__()
        self.bert             = BertModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.lstm    = CustomLSTM(
            input_size=self.bert_hidden_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_output     = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        lstm_out, _     = self.lstm(sequence_output)
        last_hidden     = lstm_out[:, -1, :]
        dropped         = self.dropout(last_hidden)
        logits          = self.fc(dropped)
        return logits
