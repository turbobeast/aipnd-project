from torch import nn, optim
import torchvision.models as models
from collections import OrderedDict

def create_classifier(inputs_num, outputs_num):
    classifier = nn.Sequential(OrderedDict([
        ('layer 1', nn.Linear(in_features=inputs_num, out_features=512)),
        ('activation 1', nn.ReLU()),
        ('dropout 1', nn.Dropout(p=0.2)),
        ('layer 2', nn.Linear(in_features=512, out_features=256)),
        ('activation 2', nn.ReLU()),
        ('dropout 2', nn.Dropout(p=0.2)),
        ('layer 3', nn.Linear(in_features=256, out_features=outputs_num)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    return classifier

def create_vgg_model(lr):
    model = models.vgg11_bn(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    classifier_in_features = 25088
    classifier = create_classifier(classifier_in_features, 102)
    model.classifier = classifier
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    return model, optimizer

def create_densenet_model(lr):
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    classifier_in_features = 1024
    classifier = create_classifier(classifier_in_features, 102)
    model.classifier = classifier
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    return model, optimizer

def create_resnet_model(lr):
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    classifier_in_features = 2048
    classifier = create_classifier(classifier_in_features, 102)
    model.fc = classifier
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    return model, optimizer

def create_alexnet_model(lr):
    model = models.alexnet(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    classifier_in_features = 9216
    classifier = create_classifier(classifier_in_features, 102)
    model.classifier = classifier
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    return model, optimizer

def setup_model(model_name, learning_rate):
    models_map = {
        "densenet": create_densenet_model,
        "vgg": create_vgg_model,
        "resnet": create_resnet_model,
        "alexnet": create_alexnet_model
    }

    model, optimizer = models_map[model_name](learning_rate)

    return model, optimizer