# dlf

DLF is a very simple deep learning framework:
- [**Tensor** class](tensor.py#L48) wraps a numpy array with the context(operation, parents) that created it
- [**topo_sort** function](tensor.py#L67) given a graph of operations return the list of computed elements
- [**backward** function](tensor.py#L84) uses operation chain rules to compute each element's gradient
- [**GD** class](optimizer.py#L14) uses gradients and learning rate to update our network
