# dlf

DLF is a very simple deep learning framework:
- [**Tensor** class](tensor.py#L48) wraps a numpy array, saving the operation that created it and the parents involved
- [**topo_sort** function](tensor.py#L62) follows parent links with a stack to walk the graph of operations
- [**backward** function](tensor.py#L79) uses the sorted nodes from topo_sort and each operation's chain rule to calculate the gradients for each node
- [**GD** class](optimizer.py#L14) uses each parameter's gradient and a learning rate to update our network
- [**Linear** class](nn.py#L6) is our layer in the multilayer.
