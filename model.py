import torch
import torch_geometric as pyg


# Reference: https://github.com/samleoqh/MSCG-Net
class SCG(torch.nn.Module):
        def __init__(self, in_ch, hidden_ch, node_size, add_diag=True, dropout=0.2):
                super(SCG, self).__init__()
                self.in_ch = in_ch
                self.hidden_ch = hidden_ch
                self.nodes = node_size[0] * node_size[1]
                self.add_diag = add_diag
                self.dropout = dropout

                # 1. reduce spatial size
                self.pooling = torch.nn.AdaptiveAvgPool2d(node_size)

                # VAE to get mu and exp(std)
                self.mu = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=in_ch, out_channels=hidden_ch, kernel_size=3, padding=1),
                        torch.nn.Dropout(dropout),
                )

                self.log_var = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=in_ch, out_channels=hidden_ch, kernel_size=1, padding=1),
                        torch.nn.Dropout(dropout),
                )


        def forward(self, feature):

                B, C, H, W = feature.size()
                nodes = self.pooling(feature)

                # VAE to get node embeddings
                mu, log_var = self.mu(nodes), self.log_var(nodes)
                z = self.reparameterize(mu, log_var)
                z = z.reshape(B, self.nodes, self.hidden_ch)

                # compute kl loss of VAE
                # KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = - (0.5*self.nodes*self.hidden_ch)* torch.mean(
                        torch.sum(1 + 2 * log_var - mu.pow(2) - log_var.exp().pow(2), 1)
                )

                # inner product to get adjacency matrix, and pass relu to restrict value of A at [0, 1]
                A = torch.matmul(z, z.permute(0, 2, 1))
                A = torch.relu(A)       # why relu instead of Sigmoid?

                # compute adaptive factor gamma
                Ad = torch.diagonal(A, dim1=1, dim2=2)
                sigma = torch.sum(Ad, dim=1)
                gamma = torch.sqrt(1 + self.nodes / (sigma + 1e-7)).unsqueeze(-1).unsqueeze(-1)

                # compute diagonal loss
                dl_loss = - gamma * torch.log(Ad+ 1.e-7).sum() / (A.size(0) * A.size(1) * A.size(2))

                A = A + gamma * torch.diag_embed(A)
                A = self.laplacian_matrix(A, self_loop=True)

                z_hat = mu.reshape(B, self.nodes, self.hidden_ch) * (1. - log_var.reshape(B, self.nodes, self.hidden_ch))
                z_hat = gamma * z_hat

                # instrad of passing embedding z to gnn, pass the spatial reduced feature map as node features
                # VAE is the part to obtain adjacency matrix
                loss = kl_loss + dl_loss

                return A, z, z_hat, loss



        def reparameterize(self, mu, std):
                z = z
                if self.training:
                        std = torch.exp(log_var / 2)
                        eps = torch.randn_like(std)
                        z = mu + std*eps
                return z


        @classmethod
        def laplacian_matrix(cls, A, self_loop=False):
                '''
                Computes normalized Laplacian matrix: A (B, N, N)
                '''
                if self_loop:
                        A = A + torch.eye(A.size(1), device=A.device).unsqueeze(0)
                # deg_inv_sqrt = (A + 1e-5).sum(dim=1).clamp(min=0.001).pow(-0.5)
                deg_inv_sqrt = (torch.sum(A, 1) + 1e-5).pow(-0.5)

                LA = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)

                return LA
        
class GCN_Layer(torch.nn.Module):
        def __init__(self, in_features, out_features, bnorm=True,
                        activation=torch.nn.ReLU(), dropout=None):
                super(GCN_Layer, self).__init__()
                self.bnorm = bnorm
                fc = [torch.nn.Linear(in_features, out_features)]
                if bnorm:
                        fc.append(BatchNorm_GCN(out_features))
                if activation is not None:
                        fc.append(activation)
                if dropout is not None:
                        fc.append(torch.nn.Dropout(dropout))
                self.fc = torch.nn.Sequential(*fc)

        def forward(self, data):
                x, A = data
                y = self.fc(torch.bmm(A, x))

                return [y, A]


def weight_xavier_init(*models):
        for model in models:
                for module in model.modules():
                        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                                # nn.init.xavier_normal_(module.weight)
                                nn.init.orthogonal_(module.weight)
                                # nn.init.kaiming_normal_(module.weight)
                                if module.bias is not None:
                                        module.bias.data.zero_()
                        elif isinstance(module, nn.BatchNorm2d):
                                module.weight.data.fill_(1)
                                module.bias.data.zero_()


class BatchNorm_GCN(torch.nn.BatchNorm1d):
        '''Batch normalization over GCN features'''

        def __init__(self, num_features):
                super(BatchNorm_GCN, self).__init__(num_features)

        def forward(self, x):
                return super(BatchNorm_GCN, self).forward(x.permute(0, 2, 1)).permute(0, 2, 1)



class SCGDecoder(torch.nn.Module):
        def __init__(self, encoder, decoder):
                super(SCGDecoder, self).__init__()
                self.encoder = SCG(2112, 1, (32, 32))
                self.decoder = torch.nn.Sequential(
                        GCN_Layer(2112, 512, bnorm=True, activation=torch.nn.ReLU(True), dropout=0.2),
                        GCN_Layer(512, 1, bnorm=False, activation=None)
                )


        def forward(self, features):
                B, C, H, W = features.size()
                A, z, z_hat, scg_loss = self.encoder(features)

                features, _ = self.decoder(features, A)
                features += z_hat

                features = torch.nn.functional.interpolate(features, (H, W), mode='bilinear')
                return features, scg_loss


class SCGNet(torch.nn.Module):
        def __init__(self, encoder, decoder):
                super(SCGNet, self).__init__()
                self.encoder = encoder
                self.decoder = decoder
        
        def forward(self, x):
                x = self.encoder(x)
                x, scg_loss = self.decoder(x)
                return x, scg_loss