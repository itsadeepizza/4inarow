# Convert models from pytorch to tflite
# pip install git+https://github.com/alibaba/TinyNeuralNetwork.git

from model.model import ConvNetNoMem, ConvNetNoGroup7
from model.model_helper import load_model
import torch


def create_tflite():
    """Convert .pth to tflite model"""
    little_last = "model_793620001.pth"
    big_no_group = "big_no_group.pth"
    player = load_model(little_last, ConvNetNoMem, device=torch.device("cpu"))
    model = player.model
    dummy_input = torch.rand(1, 6, 7)
    output_path = "./little_group.tflite"
    from tinynn.converter import TFLiteConverter
    converter = TFLiteConverter(model, dummy_input, output_path, group_conv_rewrite=True)
    converter.convert()
    # Per leggere il modello
    # https://netron.app/


def test_tflite():
    pass


if __name__ == "__main__":
    create_tflite()