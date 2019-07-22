

hidden_nodes = 36

init_nodes = 10
last_nodes = 1
layers = 1
def getweights(layers):
    weights = 0
    weights += (layers[0] - 1)*init_nodes
    weights += (layers[-1])*last_nodes
    for i in range(1,len(layers)):
        weights += (layers[i]-1)*layers[i-1]
    return weights
def getMostWeight(layer, reside_nodes, idx, most_weight, most_layer):
    layer_tmp = layer.copy()
    for i in range(reside_nodes+1):
        layer_tmp = layer.copy()
        layer_tmp[idx] += i
        if idx < len(layer) - 1:
            weights, return_layer = getMostWeight(layer_tmp, reside_nodes, idx+1, most_weight, most_layer)
            if most_weight < weights:
                most_weight = weights
                most_layer = return_layer
        else:
            layer_tmp[idx] += reside_nodes
            break
        reside_nodes -= 1
    weights = getweights(layer_tmp)
    print(layer_tmp)
    if most_weight < weights:
        most_weight = weights
        most_layer = layer_tmp
    
    return most_weight, most_layer
best = 0
blayer = []
for layer in range(1,int(hidden_nodes/2)+1):
    reside_nodes = hidden_nodes - layer*2
    Layer = [2]*layer
    weights, most_layers = getMostWeight(Layer, reside_nodes, 0, 0, [])
    if weights > best:
        best = weights
        blayer = most_layers
print(best)
print(blayer)
