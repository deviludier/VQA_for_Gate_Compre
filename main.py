# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from qiskit import QuantumCircuit, assemble, Aer
import numpy as np
import qiskit.quantum_info as qi
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from scipy.linalg import sqrtm

def density_matrix_fidelity(rho, sigma):
    '''
    Calculate the trace fidelity between two density matrix rho and sigma
    :param rho: numpy.ndarray
    :param sigma: numpy.ndarray
    :return: fidelity: float
    '''
    sz_rho = []
    sz_rho.append(rho.shape[0])
    sz_rho.append(rho.shape[1])
    if (sz_rho[0] != sigma.shape[0] or sz_rho[1] != sigma.shape[1]):
        raise Exception('Fidelity:InvalidDims','RHO and SIGMA must be matrices of the same size.')
    if (sz_rho[0] != sz_rho[1]):
        raise Exception('Fidelity:InvalidDims','RHO and SIGMA must be square.')
    sq_rho = sqrtm(rho)
    sq_prod = np.dot(np.dot(sq_rho, sigma), sq_rho)
    sq_fid = sqrtm((sq_prod +sq_prod.T)/2)
    fid = 0
    for i in range(sq_fid.shape[0]):
        fid += sq_fid[i][i].real
    return fid

def creata_ansatz1(control_params, qubit_num):
    '''
    :para qubit_num: # quantum subsystems
    :para theta: parameters in ansatz
    :return: a quantum circuit
    '''
    ansatz_circuit = QuantumCircuit(qubit_num)
    width = int(len(control_params)/2)
    if qubit_num % 2 == 0:
        for index in range(int(qubit_num/2)):
            i: int = 2*index
            ansatz_circuit.cx(i, i+1)
            ansatz_circuit.rz(control_params[i], i)
            ansatz_circuit.ry(control_params[i+width], i)
            ansatz_circuit.rz(control_params[i+1], i+1)
            ansatz_circuit.ry(control_params[i+1+width], i+1)
        for index in range(int(qubit_num/2)-1):
            i = 2*index + 1
            ansatz_circuit.cx(i, i+1)
            ansatz_circuit.rz(control_params[qubit_num + i-1], i)
            ansatz_circuit.ry(control_params[qubit_num + i-1 +width], i)
            ansatz_circuit.rz(control_params[qubit_num + i], i+1)
            ansatz_circuit.ry(control_params[qubit_num + i+width], i+1)
    if qubit_num % 2 == 1:
        for index in range(int(qubit_num / 2)):
            i: int = 2 * index
            ansatz_circuit.cx(i, i + 1)
            ansatz_circuit.rz(control_params[i], i)
            ansatz_circuit.ry(control_params[i+width], i)
            ansatz_circuit.rz(control_params[i + 1], i + 1)
            ansatz_circuit.ry(control_params[i + 1+width], i + 1)
        for index in range(int(qubit_num / 2)):
            i = 2 * index + 1
            ansatz_circuit.cx(i, i + 1)
            ansatz_circuit.rz(control_params[qubit_num + i - 2], i)
            ansatz_circuit.ry(control_params[qubit_num + i - 2+width], i)
            ansatz_circuit.rz(control_params[qubit_num + i - 1], i + 1)
            ansatz_circuit.ry(control_params[qubit_num + i - 1+width], i + 1)
    return ansatz_circuit

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # qubit_num1 = 2
    # qubit_num2 = 2
    # control_params_values1 = np.random.uniform(low=0, high=2 * np.pi, size=(qubit_num1 * 2 * 2 - 2 * 2))
    # control_params_values2 = np.random.uniform(low=0, high=2 * np.pi, size=(qubit_num2 * 2 * 2 - 2 * 2))
    # circuit1 = creata_ansatz1(control_params_values1, qubit_num1)
    # circuit2 = creata_ansatz1(control_params_values2, qubit_num2)
    #
    # # print(circuit)
    # # print(circuit1)
    # rho_1 = qi.DensityMatrix.from_instruction(circuit1)
    # rho_2 = qi.DensityMatrix.from_instruction(circuit2)
    # # print(type(rho_AB.data))
    # # print(rho_AB.data*rho_AB.data)
    # print(density_matrix_fidelity(rho_1.data, rho_2.data))

    ##############################
    # prepare initial state
    ##############################
    qubit_num1 = 2
    initial_qc1 = QuantumCircuit(qubit_num1)
    for i in range(qubit_num1):
        initial_qc1.h(i)

    qubit_num2 = 2
    initial_qc2 = QuantumCircuit(2*qubit_num2)
    for i in range(qubit_num2):
        initial_qc2.cx(i, i+qubit_num2)

    rho_1 = qi.DensityMatrix.from_instruction(initial_qc1)
    rho_2 = qi.DensityMatrix.from_instruction(initial_qc2)
    initial_rho = rho_1.tensor(rho_2)
    print(initial_rho)

    ##############################
    # Apply parameterized quantum circuit U(\theta)
    ##############################
    qubit_num3 = qubit_num1 + 2*qubit_num2
    control_params_values3 = np.random.uniform(low=0, high=2 * np.pi, size=(qubit_num3 * 2 * 2 - 2 * 2))
    control_params_values4 = np.random.uniform(low=0, high=2 * np.pi, size=(qubit_num3 * 2 * 2 - 2 * 2))
    circuit3 = creata_ansatz1(control_params_values3, qubit_num1)
    circuit4 = creata_ansatz1(control_params_values3, qubit_num1)
    circuit = QuantumCircuit(qubit_num3)
    circuit.initialize()
    circuit = circuit3 + circuit4
    sim = Aer.get_backend('aer_simulator')

