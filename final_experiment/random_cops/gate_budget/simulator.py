"""Uncompiled NumPy adjoint execution of the existing PC-QAOA circuit.

Fixed preparation circuits are evaluated once. Their Grover reflections are
applied as rank-one projectors; XY and RX operations retain the repository's
signs, angle factors, wire order, and operation order. Gate resource counts
always describe the original gate circuit, not this statevector shortcut.
"""
from pathlib import Path
import pickle
import numpy as np
import pennylane as qml

from core import constraint_handler as ch
from core import dicke_state_prep as dsp
from core.pc_qaoa import PCQAOA


class Simulator:
    def __init__(self, instance, method, database_path):
        if method not in ('pcqaoa', 'pcqaoa_stateprep'):
            raise ValueError(method)
        if method == 'pcqaoa':
            database = pickle.loads(Path(database_path).read_bytes())
            if ch.normalize_constraint(instance['vcg_constraint']) not in database:
                raise ValueError('Required trained VCG is missing; this runner never trains gadgets')
        self.model = model = PCQAOA(
            np.asarray(instance['Q']), ch.parse_constraints(instance['constraints']),
            instance['structural_indices'], instance['penalty_indices'],
            penalty_pen=float(instance['delta']), angle_strategy='QAOA', n_layers=1,
            gadget_db_path=str(Path(database_path)),
            gadget_preparation='stateprep' if method == 'pcqaoa_stateprep' else 'vcg',
            compile_on_init=False,
        )
        self.n = model.n_total
        indices = np.arange(1 << self.n)
        self.energy = np.zeros(len(indices))
        self.phase_energy = np.zeros(len(indices))
        for word, coefficient in model.problem_ham.pauli_rep.items():
            diagonal = np.ones(len(indices))
            for wire, pauli in word.items():
                if pauli != 'Z':
                    raise ValueError('Cost Hamiltonian must be diagonal')
                diagonal *= 1 - 2 * ((indices >> (self.n - 1 - wire)) & 1)
            contribution = float(np.real(coefficient)) * diagonal
            self.energy += contribution
            if word:
                self.phase_energy += contribution

        device = qml.device('default.qubit', wires=model.all_wires)

        @qml.qnode(device, interface=None)
        def initial():
            for prep in model.state_prep:
                prep.opt_circuit()
            for wire in model.slack_wires:
                qml.Hadamard(wire)
            return qml.state()

        self.initial = np.asarray(initial(), dtype=complex)
        self.operations = []
        for prep in model.mixer_preps:
            wires = list(prep.var_wires)
            if isinstance(prep, dsp.DickeStatePrep):
                if prep.mixer_type != dsp.DickeMixerType.RING_XY:
                    raise ValueError('Expected the experiment Ring-XY mixer')
                pairs = [(wires[k], wires[k+1]) for k in range(0,len(wires)-1,2)]
                pairs += [(wires[k], wires[k+1]) for k in range(1,len(wires)-1,2)]
                if len(wires) > 2:
                    pairs.append((wires[-1],wires[0]))
                for a,b in pairs:
                    ma,mb = 1 << (self.n-1-a), 1 << (self.n-1-b)
                    low = indices[((indices & ma)==0) & ((indices & mb)!=0)]
                    self.operations.append(('xy', low, low ^ ma ^ mb))
            else:
                local_device = qml.device('default.qubit', wires=wires)

                @qml.qnode(local_device, interface=None)
                def local(prep=prep):
                    prep.opt_circuit()
                    return qml.state()

                state = np.asarray(local(), dtype=complex)
                axes = wires + [w for w in model.all_wires if w not in wires]
                self.operations.append(('grover', state, axes, np.argsort(axes)))
        for wire in model.free_wires:
            mask = 1 << (self.n-1-wire)
            low = indices[(indices & mask)==0]
            self.operations.append(('rx',low,low+mask))

    def _project(self, state, operation):
        _, local, axes, undo = operation
        view = state.reshape([2]*self.n).transpose(axes).reshape(len(local),-1)
        projected = local[:,None] * (local.conj() @ view)[None,:]
        return projected.reshape([2]*self.n).transpose(undo).reshape(-1)

    def _operation(self, state, operation, beta, derivative=False):
        if operation[0] == 'grover':
            projected = self._project(state, operation)
            factor = np.exp(1j*beta/np.pi)
            return (1j/np.pi)*factor*projected if derivative else state+(factor-1)*projected
        kind, low, high = operation
        # IsingXY(2 beta): +i sin(beta); RX(beta): -i sin(beta/2).
        scale, sign = (1.,1.) if kind=='xy' else (.5,-1.)
        c, s = np.cos(scale*beta), 1j*sign*np.sin(scale*beta)
        if derivative:
            c,s = -scale*np.sin(scale*beta),1j*sign*scale*np.cos(scale*beta)
        result = np.zeros_like(state) if derivative else state.copy()
        result[low] = c*state[low]+s*state[high]
        result[high] = s*state[low]+c*state[high]
        return result

    def mixer(self, state, beta, tangent=False, reverse=False):
        derivative = np.zeros_like(state) if tangent else None
        operations = reversed(self.operations) if reverse else self.operations
        angle = -beta if reverse else beta
        for operation in operations:
            if tangent:
                derivative = (self._operation(derivative,operation,angle)
                              + self._operation(state,operation,angle,True))
            state = self._operation(state,operation,angle)
        return (state,derivative) if tangent else state

    def state(self, angles):
        state = self.initial.copy()
        for gamma,beta in np.asarray(angles).reshape(-1,2):
            state *= np.exp(-.5j*gamma*self.phase_energy)
            state = self.mixer(state,beta)
        return state

    def loss_and_gradient(self, angles):
        angles = np.asarray(angles,dtype=float).reshape(-1,2)
        state = self.initial.copy()
        records = []
        for gamma,beta in angles:
            state *= np.exp(-.5j*gamma*self.phase_energy)
            records.append(state.copy())
            state = self.mixer(state,beta)
        adjoint = self.energy*state
        loss = float(np.vdot(state,adjoint).real)
        gradient = np.zeros_like(angles)
        for layer in range(len(angles)-1,-1,-1):
            gamma,beta = angles[layer]
            _,tangent = self.mixer(records[layer],beta,tangent=True)
            gradient[layer,1] = 2*np.vdot(adjoint,tangent).real
            adjoint = self.mixer(adjoint,beta,reverse=True)
            gradient[layer,0] = 2*np.vdot(
                adjoint,(-.5j*self.phase_energy)*records[layer]).real
            adjoint *= np.exp(.5j*gamma*self.phase_energy)
        return loss,gradient

    def probabilities(self, angles):
        probabilities = abs(self.state(angles))**2
        probabilities = probabilities.reshape(1<<self.model.n_x,-1).sum(axis=1)
        return probabilities/probabilities.sum()
