# use mnist-env conda environment
import onnx
import onnxoptimizer
from qonnx.util.cleanup import cleanup
from qonnx.util.to_channels_last import to_channels_last 
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from utils import InsertSkipConnectionTranpose, RemoveTranpose
from onnx import helper


"""Transpose class for easier set up and usage"""
class InsertSkipConnectionTranpose:
    def __init__(self, model, start, end, verbose=False) -> None:
        self.model = model
        self.verbose = verbose
        self.start_node = self.find_target(start)
        self.end_node = self.find_target(end)
        assert self.start_node is not None, "Error: start node is None"
        assert self.end_node is not None, "Error: end node is None"
        
        self.end_node_replace_name = self.get_end_node_input(self.end_node)

    def get_end_node_input(self, node):
        """Find which end node input to replace with Transpose output"""
        for idx in range(len(node.input)):
            # if node does not exist replace with transpose output 
            if self._find_node(node.input[idx]) == None:  
                return node.input[idx]
            
    def _find_node(self, output_name) -> onnx.onnx_ml_pb2.NodeProto:
        """Find and return a node, given one of it's output name"""
        for node in self.model.graph.node:
            if output_name in node.output:
                return node 

    def insert_transpose(self):
        ndim = (0, 3, 1, 2)
        trans_out = f"Transpose_{self.start_node.output[0]}"

        # ChannelsLast -> ChannelsFirst transpose
        outp_trans_node = helper.make_node(
            "Transpose",                  # operation to perform 
            [self.start_node.output[0]],  # input names 
            [trans_out],                  # output names 
            perm=ndim
        )

        self.model.graph.node.append(outp_trans_node)

        for idx in range(len(self.end_node.input)):
            if self.end_node.input[idx] == self.end_node_replace_name:
                self.end_node.input[idx] = trans_out

    def find_target(self, target_name) -> onnx.onnx_ml_pb2.NodeProto:
        """Find a node given its name"""
        for node in self.model.graph.node:
            if node.name == target_name:
                if self.verbose: print('Found target node', target_name)
                return node



"""Transpose class for easier set up and usage"""
class RemoveTranpose:
    def __init__(self, model, verbose) -> None:
        self.model = model
        self.verbose = verbose
        self.nodes_update_shape = dict()

        self.transpose_nodes = self.find_targets('Transpose')  # node type  
        if self.verbose: print(f'Found {len(self.transpose_nodes)} Transposes')
        # self.end_node = self.find_target(end)
        # assert self.start_node is not None, "Error: start node is None"
        # assert self.end_node is not None, "Error: end node is None"
        
        self.subsequent_nodes = list()
        for node in self.transpose_nodes:
            self.subsequent_nodes.append(self.find_subsequent(node))

        assert len(self.transpose_nodes) == len(self.subsequent_nodes), 'Error: Not all subsequent nodes were found.'
        self.replace_transposes()
        self.update_output_shape()

    def find_subsequent(self, node):
        """Find which end node input to replace with Transpose output"""
        for idx in range(len(node.output)):
            # if node does not exist replace with transpose output 
            return self._find_node(node.output[idx])
            
    def _find_node(self, output_name) -> onnx.onnx_ml_pb2.NodeProto:
        """Find and return a node, given one of it's input names"""
        for node in self.model.graph.node:
            if output_name in node.input:
                return node 

    def replace_transposes(self):
        for t_node, s_node in zip(self.transpose_nodes, self.subsequent_nodes):
            self.model.graph.node.remove(t_node)
            replace_input_index = list(s_node.input).index(t_node.output[0])
            s_node.input[replace_input_index] = t_node.input[0]  # connect transpose input to subsequent node 
            t_node.input[0] = ''
            if 'Add' in s_node.name and s_node.name not in self.nodes_update_shape:
                self.nodes_update_shape[s_node.name] = s_node

    def update_output_shape(self):
        for node_name, s_node in self.nodes_update_shape.items():
            for value_info in self.model.graph.value_info:
                if value_info.name == s_node.output[0]:
                    tensor_type = value_info.type.tensor_type
                    # output_shape = [dim.dim_value for dim in tensor_type.shape.dim]
                    tmp = tensor_type.shape.dim[1].dim_value
                    tensor_type.shape.dim[1].ClearField('dim_value')
                    tensor_type.shape.dim[1].dim_value = tensor_type.shape.dim[-1].dim_value
                    tensor_type.shape.dim[-1].ClearField('dim_value')
                    tensor_type.shape.dim[-1].dim_value = tmp

    def remove_transpose(self, target_name):
        # Filter out the node to remove
        for idx, node in enumerate(self.model.graph.node):
            if node.name == target_name:
                self.model.graph.node[idx] = None

    def find_targets(self, target_type) -> onnx.onnx_ml_pb2.NodeProto:
        """Find all nodes of a given type"""
        nodes = list()
        for node in self.model.graph.node:
            if node.op_type == target_type and 'param' not in node.input[0] :
                if self.verbose: print('Found target node', node.name)
                nodes.append(node)
        return nodes


def prepare_hls4ml_ingestion(filename : str) -> None:
    print('Inferring shapes, renaming nodes...')
    cleanup(
        filename, 
        out_file=filename.replace('.onnx', '_clean.onnx')
    )
    
    #### Channels Last ####
    print('Converting to Channels Last...')
    to_channels_last(
        filename.replace('.onnx', '_clean.onnx'),
        out_file=filename.replace('.onnx', '_channels_last.onnx'),
    )
    
    #### Channels Last Patch ####
    channels_last_filename = filename.replace('.onnx', '_channels_last.onnx')
    model = onnx.load(channels_last_filename)

    insertObj = InsertSkipConnectionTranpose(model, 'Conv_0', 'Add_0', verbose=False)
    insertObj.insert_transpose()

    insertObj = InsertSkipConnectionTranpose(model, 'MaxPool_0', 'Add_1', verbose=False)
    insertObj.insert_transpose()

    insertObj = InsertSkipConnectionTranpose(model, 'MaxPool_1', 'Add_2', verbose=False)
    insertObj.insert_transpose()

    onnx.save(insertObj.model, filename.replace('.onnx', '_skip_conn.onnx'))
    
    model = onnx.load(filename.replace('.onnx', '_skip_conn.onnx'))
    removeTransposeObj = RemoveTranpose(model, verbose=False)
    onnx.save(
        removeTransposeObj.model, 
        filename.replace('.onnx', '_no_transpose.onnx')
    )
    
    cleanup(
        filename.replace('.onnx', '_no_transpose.onnx'),
        out_file=filename.replace('.onnx', '_clean2.onnx')
    )
    
    #### Convert GEMM to MatMul/Add ####
    print('Converting GEMM to MatMul...')
    model = ModelWrapper(filename.replace('.onnx', '_clean2.onnx'))

    transformation = GemmToMatMul()

    model_changed = True
    while model_changed:
        model, model_changed = transformation.apply(model)

    onnx.save(
        model.model, 
        filename.replace('.onnx', '_gemm_to_matmul.onnx')
    )
    
    #### Final cleanup ####
    cleanup(
        filename.replace('.onnx', '_gemm_to_matmul.onnx'),
        out_file=filename.replace('.onnx', '_final.onnx')
    )
    print('Done.')
