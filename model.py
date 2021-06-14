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

                self.sigmoid = torch.nn.Sigmoid()
                self.relu = torch.nn.ReLU(inplace=False)

                # 1. reduce spatial size
                self.pooling = torch.nn.AdaptiveAvgPool2d(node_size)

                # VAE to get mu and exp(std)
                self.mu = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=in_ch, out_channels=hidden_ch, kernel_size=3, padding=1),
                        torch.nn.Dropout(dropout),
                )

                self.log_var = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=in_ch, out_channels=hidden_ch, kernel_size=1, padding=0),
                        torch.nn.Dropout(dropout),
                )


        def forward(self, feature):

                B, C, H, W = feature.size()
                nodes = self.pooling(feature)

                # VAE to get node embeddings
                mu, log_var = self.mu(nodes), self.log_var(nodes)
                z = self.reparameterize(mu, log_var)
                z = z.reshape(B, self.nodes, self.hidden_ch)
                # print(torch.min(z), torch.max(z))

                # compute kl loss of VAE
                # KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = - (0.5*self.nodes*self.hidden_ch)* torch.mean(
                        torch.sum(1 + 2 * log_var - mu.pow(2) - log_var.exp().pow(2), 1)
                )

                # inner product to get adjacency matrix, and pass relu to restrict value of A at [0, 1]
                A = torch.matmul(z, z.permute(0, 2, 1))
                A = self.relu(A)       # why relu instead of Sigmoid?
                # A = self.sigmoid(A)

                # print(torch.min(A), torch.max(A))

                # compute adaptive factor gamma
                Ad = torch.diagonal(A, dim1=1, dim2=2)
                sigma = torch.sum(Ad, dim=1)
                gamma = torch.sqrt(1 + self.nodes / (sigma + 1e-7)).unsqueeze(-1).unsqueeze(-1)

                # compute diagonal loss
                dl_loss = - gamma * torch.log(Ad+ 1.e-7).sum() / (A.size(0) * A.size(1) * A.size(2))

                A_diag = []
                for i in range(B):
                        # iterate each batch to get diagonal
                        A_diag.append(torch.diagflat(Ad[i]))
                
                A = A + gamma * torch.stack(A_diag)
                A = self.laplacian_matrix(A, self_loop=True)

                z_hat = mu.reshape(B, self.nodes, self.hidden_ch) * (1. - log_var.reshape(B, self.nodes, self.hidden_ch))
                z_hat = gamma * z_hat

                # instrad of passing embedding z to gnn, pass the spatial reduced feature map as node features
                # VAE is the part to obtain adjacency matrix
                # loss = kl_loss + dl_loss

                return A, nodes, z_hat, kl_loss, dl_loss



        def reparameterize(self, mu, std):
                z = mu
                if self.training:
                        std = torch.exp(std/2)
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
        
class GCNLayer(torch.nn.Module):
        def __init__(self, in_features, out_features, bnorm=True,
                        activation=torch.nn.ReLU(), dropout=None):
                super(GCNLayer, self).__init__()
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
                        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                                # nn.init.xavier_normal_(module.weight)
                                torch.nn.init.orthogonal_(module.weight)
                                # nn.init.kaiming_normal_(module.weight)
                                if module.bias is not None:
                                        module.bias.data.zero_()
                        elif isinstance(module, torch.nn.BatchNorm2d):
                                module.weight.data.fill_(1)
                                module.bias.data.zero_()


class BatchNorm_GCN(torch.nn.BatchNorm1d):
        '''Batch normalization over GCN features'''

        def __init__(self, num_features):
                super(BatchNorm_GCN, self).__init__(num_features)

        def forward(self, x):
                return super(BatchNorm_GCN, self).forward(x.permute(0, 2, 1)).permute(0, 2, 1)



class SCGDecoder(torch.nn.Module):
        def __init__(self, encoder, decoder, activation=None, scale_size=None, scale_factor=None):
                super(SCGDecoder, self).__init__()
                self.encoder = SCG(2112, 1, (32, 32))
                self.gcn_1 = GCNLayer(2112, 512, bnorm=True, activation=torch.nn.ReLU(inplace=False), dropout=0.2)
                self.gcn_2 = GCNLayer(512, 1, bnorm=False, activation=None)
                
                self.scale_size = scale_size
                self.scale_factor = scale_factor
                self.activation = activation

        def forward(self, features):
                A, z, z_hat, kl_loss, dl_loss = self.encoder(features)
                B, C, H, W = z.size()
                # print(features.size(), A.size())
                features, A = self.gcn_1([z.reshape(B, -1, C), A])
                features, _ = self.gcn_2([features, A])
                # print(torch.where(features == 0))
                features += z_hat
                # features += 1e-7

                features = features.reshape(B, 1, H, W)
                if self.scale_size:
                        features = torch.nn.functional.interpolate(features, self.scale_size, mode='bilinear', align_corners=False)
                elif self.scale_factor:
                        features = torch.nn.functional.interpolate(features, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
                
                if self.activation:
                        features = self.activation(features)
                return features, kl_loss, dl_loss




class GCNBlock(torch.nn.Module):
        def __init__(self, in_ch, hidden_ch, out_ch, scale_size=None, scale_factor=None, activation=None, dropout=None):
                super(GCNBlock, self).__init__()
                self.gcn_1 = GCNLayer(in_ch, hidden_ch, bnorm=True, activation=activation, dropout=dropout)
                self.gcn_2 = GCNLayer(hidden_ch, out_ch, bnorm=False, activation=None)

                self.scale_size = scale_size
                self.scale_factor = scale_factor
                self.activation = activation
                
        def forward(self, feature):
                feature, A, residual = feature
                B, C, H, W = feature.size()
                feature = self.gcn_1([feature.reshape(B, -1, C), A])
                feature, A = self.gcn_2(feature)

                feature = feature.reshape(B, -1, H, W)
                residual = residual.reshape(B, -1, H, W)
                feature += residual
                if self.scale_size:
                        feature = torch.nn.functional.interpolate(feature, self.scale_size, mode='bilinear', align_corners=False)
                        # scale_factor = (self.scale_size[0]/H, self.scale_size[1]/W)
                        # A =  torch.nn.functional.interpolate(A, scale_factor=scale_factor, mode='bilinear', align_corners=False)
                elif self.scale_factor:
                        feature = torch.nn.functional.interpolate(feature, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
                        A = torch.nn.functional.interpolate(A, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

                if self.activation:
                        feature = self.activation(feature)

                return feature


class SCGraphUnetDecoder(torch.nn.Module):
        def __init__(self, node_sizes, in_channels, out_channels, device):
                super(SCGraphUnetDecoder, self).__init__()
                self.encoders = torch.nn.ModuleList([
                        SCG(96, 1, (64, 64)).to(device),
                        SCG(384, 1, (64, 64)).to(device),
                        SCG(768, 1, (64, 64)).to(device),
                        SCG(2112, 1, (32, 32)).to(device),
                        # SCG(2208, 1, (16, 16)).to(device)
                ])
                # for node_size, in_ch, out_ch in zip(node_sizes, in_channels, out_channels):
                #         self.encoders.append(SCG(in_ch, node_size, out_ch))
                self.decoders = torch.nn.ModuleList([
                        GCNBlock(in_ch=96, hidden_ch=64, out_ch=1, scale_size=(512, 512), dropout=0.2).to(device),
                        GCNBlock(in_ch=384, hidden_ch=128, out_ch=1, scale_size=(512, 512), dropout=0.2).to(device),
                        GCNBlock(in_ch=768, hidden_ch=256, out_ch=1, scale_size=(512, 512), dropout=0.2).to(device),
                        GCNBlock(in_ch=2112, hidden_ch=512, out_ch=1, scale_size=(512, 512), dropout=0.2).to(device),
                        # GCNBlock(in_ch=2208, hidden_ch=1024, out_ch=1, scale_size=(512, 512), dropout=0.2).to(device),
                ])

                self.conv1 =  torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, padding=0)

        def forward(self, features):
                features = features[1:-1]

                assert len(features) == len(self.encoders), "Length of input must be equal to encoders"

                kl_losses, dl_losses = [], []

                feats = None
                for feature, encoder, decoder in zip(features, self.encoders, self.decoders):
                        A, z, z_hat, kl_loss, dl_loss = encoder(feature)
                  
                        B, C, H, W = z.size()
                        feat = decoder([z, A, z_hat])
                        # feat += z_hat
                        kl_losses.append(kl_loss)
                        dl_losses.append(dl_loss)

                        if feats is not None:
                                feats = torch.cat((feats, feat), dim=1)
                        else:
                                feats = feat

                # B, C, H, W = feats.size()
                # feat = torch.mean(feats.view(B, H, W, C), -1)

                feats = self.conv1(feats)
                # feat = feat.unsqueeze(1)
                feats = torch.nn.Sigmoid()(feats)

                return feats, torch.mean(torch.stack(kl_losses)), torch.mean(torch.stack(dl_losses))



class SCGraphUnet(torch.nn.Module):
        def __init__(self, encoder, decoder):
                super(SCGraphUnet, self).__init__()
                self.encoder = encoder
                self.decoder = decoder

                # weight_xavier_init(self.encoder, self.decoder)
        
        def forward(self, x):
                x = self.encoder(x)
                x, kl_loss, dl_loss = self.decoder(x)
                return x, kl_loss, dl_loss



class SCGNet(torch.nn.Module):
        def __init__(self, encoder, decoder):
                super(SCGNet, self).__init__()
                self.encoder = encoder
                self.decoder = decoder

                # weight_xavier_init(self.encoder, self.decoder)
        
        def forward(self, x):
                x = self.encoder(x)[-2]
                x, kl_loss, dl_loss = self.decoder(x)
                return x, kl_loss, dl_loss



