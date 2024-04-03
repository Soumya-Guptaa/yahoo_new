import torch
import torch
import torch.nn as nn
import torch.nn.functional as F


# Remove positional encoding and perm invariant network

class hbiascorrect(nn.Module):

    def __init__(self,seq_len, seq_dim,g_embed_dim,f_embed_dim, output_dim,num_layers, num_heads, hidden_size, dropout_rate):

        super(hbiascorrect, self).__init__()

        self.seq_len = seq_len
        self.seq_dim = seq_dim 
        self.output_dim = output_dim 
        self.gru_latent_dim = f_embed_dim+output_dim

        self.matlayer_left = nn.Sequential(
                        nn.Linear(seq_dim, seq_dim),
                        nn.ReLU(),
                        nn.Linear(seq_dim,seq_dim)
                        )
        
        self.matlayer_right = nn.Sequential(
                        nn.Linear(seq_dim, seq_dim),
                        nn.ReLU(),
                        nn.Linear(seq_dim,seq_dim)
                        )

        self.gembed = nn.Linear(seq_dim,g_embed_dim) 
        self.fembed = nn.Linear(seq_dim, f_embed_dim)

        g_encoder_layer = nn.TransformerEncoderLayer(d_model= g_embed_dim+seq_len, nhead= num_heads)
        f_encoder_layer = nn.TransformerEncoderLayer(d_model= f_embed_dim+output_dim, nhead= num_heads)
        
        self.gtransformer = nn.TransformerEncoder(g_encoder_layer, num_layers)
        self.ftransformer = nn.TransformerEncoder(f_encoder_layer, num_layers)

        self.gpredict = nn.Sequential(
                                    nn.Linear( g_embed_dim+seq_len, output_dim),
                                    nn.ReLU(),
                                    nn.Linear( output_dim, output_dim),
                                    nn.Sigmoid()
                                    )
        
        # self.fpredict = nn.Sequential(nn.Linear( f_embed_dim+output_dim+seq_len,output_dim),
        #                             nn.ReLU())
        
        self.phi = nn.Sequential(
                                nn.Linear( f_embed_dim+output_dim+seq_len, output_dim),
                                nn.ReLU(),
                                nn.Linear( output_dim, output_dim)
                                )

        self.rho = nn.Sequential(
                                nn.Linear( output_dim, output_dim),
                                nn.ReLU(),
                                nn.Linear( output_dim, output_dim)
                                )

        self.gru = nn.GRU(input_size= seq_dim+self.gru_latent_dim,hidden_size= hidden_size,num_layers= num_layers, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)

        # Add this line to enable nested tensors
        #self.use_nested_tensor = True
        

    def forward(self, x):

        # print(f"x.shape: {x.shape}")

        z_left = self.matlayer_left(x)
        z_right = self.matlayer_right(x)

        batch_len = x.size()[0]
        
        z_right_transposed_tensor = z_right.transpose(1, 2)
        zmat = torch.matmul(z_left, z_right_transposed_tensor)
        
        temp = 0.1
        omega = 0.5

        seed_mat = zmat 

        permmat = self.pytorch_sinkhorn_iters_mask(seed_mat, temp, noise_factor=1.0, n_iters=20)
        
        x_wbias = torch.matmul(permmat, x)

        x_gemb = self.gembed(x)
        gtx_in = self.make_sequence_for_Gtransformer(x_gemb)
        gtx_out = self.gtransformer(gtx_in)
        yhat = self.gpredict(gtx_out)

        x_femb = self.fembed(x)
        ftx_in = torch.cat((x_femb, yhat),dim=-1)
        ftx_out = self.ftransformer(ftx_in)
        gru_latent = torch.sum(ftx_out, dim=1)

        # phi_in= ftx_out[:,self.seq_len,:]
        # phi_out = self.phi(phi_in)
        
        # rho_seq= torch.reshape(phi_out,(batch_len, self.seq_len, self.output_dim))
        # rho_in = torch.sum(rho_seq,dim=1)
        # rho_out = self.rho(rho_in)
        
        
        gru_latent_repeat = gru_latent.repeat_interleave(self.seq_len).reshape((-1,self.seq_len,self.gru_latent_dim))

        # print("x.size()")
        # print(x.size())

        # print("gru_latent.size()")
        # print(gru_latent.size())

        # print("gru_latent_repeat.size()")
        # print(gru_latent_repeat.size())

        gru_in = torch.cat((x,gru_latent_repeat),dim=-1)
        gru_out,_ = self.gru(gru_in) 
        yhat_out = self.fc(gru_out)



        x_gemb_wbias = self.gembed(x_wbias)
        gtx_in_wbias = self.make_sequence_for_Gtransformer(x_gemb_wbias)
        gtx_out_wbias = self.gtransformer(gtx_in_wbias)
        yhat_wbias = self.gpredict(gtx_out_wbias)

        x_femb_wbias = self.fembed(x_wbias)        
        ftx_in_wbias = torch.cat((x_femb_wbias, yhat_wbias), dim = -1)               
        ftx_out_wbias = self.ftransformer(ftx_in_wbias)
        gru_latent_wbias = torch.sum(ftx_out_wbias, dim=1)

        # phi_in_wbias= ftx_out_wbias[:,self.seq_len,:]
        # phi_out_wbias = self.phi(phi_in_wbias)

        # rho_seq_wbias= torch.reshape(phi_out_wbias,(batch_len, self.seq_len, self.output_dim))
        # rho_in_wbias = torch.sum(rho_seq_wbias,dim=1)
        # rho_out_wbias = self.rho(rho_in_wbias)
        
        
        gru_latent_repeat_wbias= gru_latent_wbias.repeat_interleave(self.seq_len).reshape((-1,self.seq_len,self.gru_latent_dim))
        
        gru_in_wbias = torch.cat((x,gru_latent_repeat_wbias),dim=-1)
        gru_out_wbias,_ = self.gru(gru_in_wbias)
        yhat_out_wbias = self.fc(gru_out_wbias)


        return yhat_out, yhat_out_wbias, yhat


#---------------------------------------------------------------------------
    def gen_ftx_in(self,X_orig, Y):

        device = X_orig.device
        X = torch.cat((X_orig,Y), dim=-1)

        batch_len, num_entry, dim_X= X.size()
        

        id_mat = torch.eye(num_entry, device = device).expand(batch_len,num_entry,num_entry)


        X = torch.cat((X,id_mat), dim= -1)

        # This tensor will hold the expanded X
        result_tensor = torch.empty(num_entry, batch_len, num_entry + 1, dim_X + num_entry, device = device)
        
        for idx in range(num_entry):

            # Extract idx_th entries from each batch
            seq_entry = X[:,idx,:].clone().unsqueeze(1)

            # Insert ones at corresponding positional encoded columns
            seq_entry[ :, :, dim_X:] = torch.ones_like(seq_entry[ :, :, dim_X:])
            
            # Concatenate at num_entry_th row for each batch
            temp_X = torch.cat((X, seq_entry), dim=1)

            # Store the value of temp_X in the result_tensor
            result_tensor[idx, :, :, :] = temp_X
            
        result_tensor = torch.cat([result_tensor[i,:,:,:] for i in range(result_tensor.size()[0])], dim=1)

        result_tensor = result_tensor.view(-1, num_entry+1, dim_X + num_entry)
            
        return result_tensor

#------------------------------------------------------------------------------

    def ftransformer_processing(self,input_seq,output):
        device= input_seq
        seq_len = input_seq.size()[1]
        batch_size = input_seq.size()[0]
        
        new_sequence = []
        for inps in input_seq:
            for entry in inps:
                z = entry.clone()
                z[ seq_len+1 : ] = 1
                z = z.unsqueeze(dim=0)
                z = torch.cat((inps, z), dim=-2)
                new_sequence.append(z)
        new_sequence_tensor = torch.stack(new_sequence)

     
        return new_sequence_tensor
       
    
    def make_sequence_for_Gtransformer(self, input):
        device= input.device
        seq_len = input.size()[1]
        batch_size = input.size()[0]
        identity_matrix = torch.eye(seq_len, device = device).unsqueeze(0).expand(batch_size, -1, -1)
        sequence = torch.cat((input,identity_matrix),dim=-1)
        return sequence


    def make_sequence_for_Ftransformer(self, input ,output):
        device= input.device
        seq_len = input.size()[1]
        batch_size = input.size()[0]
        identity_matrix = torch.eye(seq_len, device = device).unsqueeze(0).expand(batch_size, -1, -1)
        sequence = torch.cat((input,output , identity_matrix),dim=-1)
        
#        print("input.size()", input.size())
#        print("output.size()", output.size())
#        print('sequence.shape')
#        print(sequence.shape)
#        print("ok")

        
        return sequence


    def sample_gumbel(self, shape, device):
        U = torch.rand(shape).float().to(device)
        
        self.eps = 1
        return -torch.log(self.eps - torch.log(U + self.eps))

    def pytorch_sinkhorn_iters_mask(self, log_alpha, temp, noise_factor=1.0, n_iters=20):
        device = log_alpha.device  # Get the device of the input tensor
        batch, nt = log_alpha.size(0), log_alpha.size(1)
        log_alpha = log_alpha.view(-1, nt, nt)
        noise = self.sample_gumbel((batch, nt, nt), device=device) * noise_factor

        log_alpha = log_alpha + noise
        log_alpha = log_alpha / temp

        for _ in range(n_iters):
            log_alpha_copy = log_alpha.clone()  # Create a copy of log_alpha
            log_alpha_copy -= torch.logsumexp(log_alpha, dim=2, keepdim=True)
            z = log_alpha_copy.clone()
            log_alpha_copy -= torch.logsumexp(z, dim=1, keepdim=True)
            log_alpha = log_alpha_copy  # Assign the updated value back to log_alpha

        return torch.exp(log_alpha)


class Permscore(nn.Module):
  def __init__(self, input_size, hidden_size, output_dim):
        super(Permscore, self).__init__()
        
        self.hidden_size = hidden_size
        self.fc = nn.Linear(input_size, 1)
        

  def forward(self, x):
        
        out = self.fc(x)
        
        return out



class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out)
        return out


