import torch
from torch import nn
from deeplabv3p import SeperateConv, Encoder, Decoder, Deeplabv3p
from torchsummary import summary

def test_SeperateConv():
    x = torch.randn((1, 3, 50, 50))

    sep1 = SeperateConv(3, 7, 3, dilation=1)
    sep1.eval()
    y = sep1(x)
    assert y.shape[1:] == torch.Size([7, 50, 50])

    sep2 = SeperateConv(3, 7, 3, dilation=2)
    sep2.eval()
    y = sep2(x)
    assert y.shape[1:] == torch.Size([7, 50, 50])

def test_Encoder():
    encoder = Encoder(in_channels=1, out_channels=1)
    x = torch.randn((1, 1, 50, 50))
    encoder.eval()
    y = encoder(x)
    assert x.shape == y.shape

def test_Decoder():
    sc = torch.randn((1, 3, 40, 40))
    enc = torch.randn((1, 3, 10, 10))
    decoder = Decoder(sc_in=3, sc_out=5, enc_in=3, out_channels=10)
    decoder.eval()
    y = decoder(enc, sc)
    assert y.shape[-2:] == sc.shape[-2:]
    assert y.shape[1] == 10

def test_deeplabv3():
    model = Deeplabv3p()
    x = torch.randn((1, 3, 256, 256))
    model.eval()
    summary(model, (3, 256, 256))
    y = model(x)
    print(x.shape)
    print(y.shape)

if __name__ == '__main__':
    # test_SeperateConv()
    # test_Encoder()
    # test_Decoder()
    test_deeplabv3()