# -*- coding: utf-8 -*-
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2014 and later, Alexander J G Pitchford
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

# @author: Alexander Pitchford
# @email1: agp1@aber.ac.uk
# @email2: alex.pitchford@gmail.com
# @organization: Aberystwyth University
# @supervisor: Daniel Burgarth

"""
Classes that define the dynamics of the (quantum) system and target evolution
to be optimised.
The contols are also defined here, i.e. the dynamics generators (Hamiltonians,
Limbladians etc). The dynamics for the time slices are calculated here, along
with the evolution as determined by the control amplitudes.

See the subclass descriptions and choose the appropriate class for the
application. The choice depends on the type of matrix used to define
the dynamics.

These class implement functions for getting the dynamics generators for
the combined (drift + ctrls) dynamics with the approriate operator applied

Note the methods in these classes were inspired by:
DYNAMO - Dynamic Framework for Quantum Optimal Control
See Machnes et.al., arXiv.1011.4874
"""
import os
import warnings
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
# QuTiP
from qutip import Qobj
from qutip.sparse import sp_eigs, _dense_eigs
import qutip.settings as settings
# QuTiP logging
import qutip.logging_utils as logging
logger = logging.get_logger()
# QuTiP control modules
import qutip.control.errors as errors
import qutip.control.tslotcomp as tslotcomp
import qutip.control.fidcomp as fidcomp
import qutip.control.propcomp as propcomp
import qutip.control.symplectic as sympl
import qutip.control.dump as qtrldump

DEF_NUM_TSLOTS = 10
DEF_EVO_TIME = 1.0

def _is_string(var):
    try:
        if isinstance(var, basestring):
            return True
    except NameError:
        try:
            if isinstance(var, str):
                return True
        except:
            return False
    except:
        return False

    return False

def _check_ensemble_size(ne):
    try:
        ne = int(ne)
    except:
        raise ValueError("ensemble_size must be positive integer")
    if ne < 1:
        raise ValueError("ensemble_size must be positive integer")

    return ne

#def _check_ctrls_container(ctrls):
#    """
#    Check through the controls container.
#    Convert to an array if its a list of lists
#    return the processed container
#    raise type error if the container structure is invalid
#    """
#    if isinstance(ctrls, (list, tuple)):
#        # Check to see if list of lists
#        try:
#            if isinstance(ctrls[0], (list, tuple)):
#                ctrls = np.array(ctrls)
#        except:
#            pass
#
#    if isinstance(ctrls, np.ndarray):
#        if len(ctrls.shape) != 2:
#            raise TypeError("Incorrect shape for ctrl dyn gen array")
#        for k in range(ctrls.shape[0]):
#            for j in range(ctrls.shape[1]):
#                if not isinstance(ctrls[k, j], Qobj):
#                    raise TypeError("All control dyn gen must be Qobj")
#    elif isinstance(ctrls, (list, tuple)):
#        for ctrl in ctrls:
#            if not isinstance(ctrl, Qobj):
#                raise TypeError("All control dyn gen must be Qobj")
#    else:
#        raise TypeError("Controls list or array not set correctly")
#
#    return ctrls

def _check_nested_Qobj(chk, attr_name, min_nest=0, max_nest=1, nest_level=0):
    if min_nest == 0:
        msg = ("{} should be Qobj or nested list of Qobj, "
               "up to {} nest levels".format(attr_name, max_nest))
    else:
        msg = ("{} should be nested list of Qobj, with min {} and max {} "
               "nest levels".format(attr_name, min_nest, max_nest))

    if isinstance(chk, Qobj):
        if nest_level < min_nest:
            raise TypeError(msg)
    else:
        if isinstance(chk, (list, tuple)):
            if nest_level > max_nest:
                raise TypeError("{} is nested beyond the "
                                "maximum {}".format(attr_name, max_nest))
            for o in chk:
                _check_nested_Qobj(o, attr_name, min_nest=min_nest,
                                  max_nest=max_nest, nest_level=nest_level+1)
        else:
            raise TypeError(msg)

def _check_initial(initial):
    _check_nested_Qobj(initial, 'initial', max_nest=1)

def _check_target(target):
    _check_nested_Qobj(target, 'target', max_nest=1)

def _check_drift(drift):
    _check_nested_Qobj(drift, 'drift', max_nest=2)

def _check_ctrls(ctrls):
    _check_nested_Qobj(ctrls, 'ctrls', min_nest=1, max_nest=3)

def _create_nested_dense(ql):
    """
    created a nested list of ndarray from nested list of Qobj
    """
    if isinstance(ql, Qobj):
        return ql.full()
    elif isinstance(ql, (list, tuple)):
        return [_create_nested_dense(q) for q in ql]
    else:
        raise TypeError("Unexpected type {}".format(type(ql)))

warnings.simplefilter('always', DeprecationWarning) #turn off filter
def _attrib_deprecation(message, stacklevel=3):
    """
    Issue deprecation warning
    Using stacklevel=3 will ensure message refers the function
    calling with the deprecated parameter,
    """
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)

def _func_deprecation(message, stacklevel=3):
    """
    Issue deprecation warning
    Using stacklevel=3 will ensure message refers the function
    calling with the deprecated parameter,
    """
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)

class Dynamics(object):
    """
    This is a base class only. See subclass descriptions and choose an
    appropriate one for the application.

    Note that initialize_controls must be called before most of the methods
    can be used. init_timeslots can be called sometimes earlier in order
    to access timeslot related attributes

    This acts as a container for the operators that are used to calculate
    time evolution of the system under study. That is the dynamics generators
    (Hamiltonians, Lindbladians etc), the propagators from one timeslot to
    the next, and the evolution operators. Due to the large number of matrix
    additions and multiplications, for small systems at least, the optimisation
    performance is much better using ndarrays to represent these operators.
    However

    Attributes
    ----------
    log_level : integer
        level of messaging output from the logger.
        Options are attributes of qutip.logging_utils,
        in decreasing levels of messaging, are:
        DEBUG_INTENSE, DEBUG_VERBOSE, DEBUG, INFO, WARN, ERROR, CRITICAL
        Anything WARN or above is effectively 'quiet' execution,
        assuming everything runs as expected.
        The default NOTSET implies that the level will be taken from
        the QuTiP settings file, which by default is WARN

    params:  Dictionary
        The key value pairs are the attribute name and value
        Note: attributes are created if they do not exist already,
        and are overwritten if they do.

    stats : Stats
        Attributes of which give performance stats for the optimisation
        set to None to reduce overhead of calculating stats.
        Note it is (usually) shared with the Optimizer object

    tslot_computer : TimeslotComputer (subclass instance)
        Used to manage when the timeslot dynamics
        generators, propagators, gradients etc are updated

    prop_computer : PropagatorComputer (subclass instance)
        Used to compute the propagators and their gradients

    fid_computer : FidelityComputer (subclass instance)
        Used to computer the fidelity error and the fidelity error
        gradient.

    ensemble_size : int
        Number of systems in the ensemble. Default is 1.

    memory_optimization : int
        Level of memory optimisation. Setting to 0 (default) means that
        execution speed is prioritized over memory.
        Setting to 1 means that some memory prioritisation steps will be
        taken, for instance using Qobj (and hence sparse arrays) as the
        the internal operator data type, and not caching some operators
        Potentially further memory saving maybe made with
        memory_optimization > 1.
        The options are processed in _set_memory_optimizations, see
        this for more information. Individual memory saving  options can be
        switched by settting them directly (see below)

    oper_dtype : type
        Data type for internal dynamics generators, propagators and time
        evolution operators. This can be ndarray or Qobj, or (in theory) any
        other representaion that supports typical matrix methods (e.g. dot)
        ndarray performs best for smaller quantum systems.
        Qobj may perform better for larger systems, and will also
        perform better when (custom) fidelity measures use Qobj methods
        such as partial trace.
        See _choose_oper_dtype for how this is chosen when not specified

    cache_phased_dyn_gen : bool
        If True then the dynamics generators will be saved with and
        without the propagation prefactor (if there is one)
        Defaults to True when memory_optimization=0, otherwise False

    cache_prop_grad : bool
        If the True then the propagator gradients (for exact gradients) will
        be computed when the propagator are computed and cache until
        the are used by the fidelity computer. If False then the
        fidelity computer will calculate them as needed.
        Defaults to True when memory_optimization=0, otherwise False

    cache_dyn_gen_eigenvectors_adj: bool
        If True then DynamicsUnitary will cached the adjoint of
        the Hamiltion eignvector matrix
        Defaults to True when memory_optimization=0, otherwise False

    sparse_eigen_decomp: bool
        If True then DynamicsUnitary will use the sparse eigenvalue
        decomposition.
        Defaults to True when memory_optimization<=1, otherwise False

    num_tslots : integer
        Number of timeslots (aka timeslices)

    num_ctrls : integer
        Number of controls.
        Note this is calculated as the length of ctrl_dyn_gen when first used.
        And is recalculated during initialise_controls only.

    evo_time : float
        Total time for the evolution

    tau : array[num_tslots] of float
        Duration of each timeslot
        Note that if this is set before initialize_controls is called
        then num_tslots and evo_time are calculated from tau, otherwise
        tau is generated from num_tslots and evo_time, that is
        equal size time slices

    time : array[num_tslots+1] of float
        Cumulative time for the evolution, that is the time at the start
        of each time slice

    drift_dyn_gen : Qobj or list of Qobj
        Drift or system dynamics generator (Hamiltonian)
        Matrix defining the underlying dynamics of the system
        Can also be a list of Qobj (length num_tslots) for time varying
        drift dynamics

    ctrl_dyn_gen : List of Qobj
        Control dynamics generator (Hamiltonians)
        List of matrices defining the control dynamics

    initial : Qobj
        Starting state / gate
        The matrix giving the initial state / gate, i.e. at time 0
        Typically the identity for gate evolution

    target : Qobj
        Target state / gate:
        The matrix giving the desired state / gate for the evolution

    ctrl_amps : array[num_tslots, num_ctrls] of float
        Control amplitudes
        The amplitude (scale factor) for each control in each timeslot

    initial_ctrl_scaling : float
        Scale factor applied to be applied the control amplitudes
        when they are initialised
        This is used by the PulseGens rather than in any fucntions in
        this class

    initial_ctrl_offset  : float
        Linear offset applied to be applied the control amplitudes
        when they are initialised
        This is used by the PulseGens rather than in any fucntions in
        this class

    dyn_gen : List of Qobj
        Dynamics generators
        the combined drift and control dynamics generators
        for each timeslot

    prop : list of Qobj
        Propagators - used to calculate time evolution from one
        timeslot to the next

    prop_grad : array[num_tslots, num_ctrls] of Qobj
        Propagator gradient (exact gradients only)
        Array  of Qobj that give the gradient
        with respect to the control amplitudes in a timeslot
        Note this attribute is only created when the selected
        PropagatorComputer is an exact gradient type.

    fwd_evo : List of Qobj
        Forward evolution (or propagation)
        the time evolution operator from the initial state / gate to the
        specified timeslot as generated by the dyn_gen

    onwd_evo : List of Qobj
        Onward evolution (or propagation)
        the time evolution operator from the specified timeslot to
        end of the evolution time as generated by the dyn_gen

    onto_evo : List of Qobj
        'Backward' List of Qobj propagation
        the overlap of the onward propagation with the inverse of the
        target.
        Note this is only used (so far) by the unitary dynamics fidelity

    evo_current : Boolean
        Used to flag that the dynamics used to calculate the evolution
        operators is current. It is set to False when the amplitudes
        change

    fact_mat_round_prec : float
        Rounding precision used when calculating the factor matrix
        to determine if two eigenvalues are equivalent
        Only used when the PropagatorComputer uses diagonalisation

    def_amps_fname : string
        Default name for the output used when save_amps is called

    unitarity_check_level : int
        If > 0 then unitarity of the system evolution is checked at at
        evolution recomputation.
        level 1 checks all propagators
        level 2 checks eigen basis as well
        Default is 0

    unitarity_tol :
        Tolerance used in checking if operator is unitary
        Default is 1e-10

    dump : :class:`dump.DynamicsDump`
        Store of historical calculation data.
        Set to None (Default) for no storing of historical data
        Use dumping property to set level of data dumping

    dumping : string
        level of data dumping: NONE, SUMMARY, FULL or CUSTOM
        See property docstring for details

    dump_to_file : bool
        If set True then data will be dumped to file during the calculations
        dumping will be set to SUMMARY during init_evo if dump_to_file is True
        and dumping not set.
        Default is False

    dump_dir : string
        Basically a link to dump.dump_dir. Exists so that it can be set through
        dyn_params.
        If dump is None then will return None or will set dumping to SUMMARY
        when setting a path

    """
    def __init__(self, optimconfig, params=None):
        self.config = optimconfig
        self.params = params
        self.reset()

    def reset(self):
        # Link to optimiser object if self is linked to one
        self.parent = None
        # Main functional attributes
        self.time = None
        self.initial = None
        self.target = None
        self.ctrl_amps = None
        self.initial_ctrl_scaling = 1.0
        self.initial_ctrl_offset = 0.0
        self.drift_dyn_gen = None
        self.ctrl_dyn_gen = None
        self._tau = None
        self._evo_time = None
        self._num_ctrls = None
        self._num_tslots = None

        # ensemble attributes
        self.ensemble_size = 1
        self._e_shares_initial = True
        self._e_shares_drift = True
        self._e_shares_ctrls = True
        self._e_shares_target = True

        # attributes used for processing evolution
        self.memory_optimization = 0
        self.oper_dtype = None
        self.cache_phased_dyn_gen = None
        self.cache_prop_grad = None
        self.cache_dyn_gen_eigenvectors_adj = None
        self.sparse_eigen_decomp = None
        self._dyn_dims = None
        self._dyn_shape = None
        self._sys_dims = None
        self._sys_shape = None
        self._time_depend_drift = [False]
        self._time_depend_ctrl_dyn_gen = [False]
        # These internal attributes will be of the internal operator data type
        # used to compute the evolution
        # Note this maybe ndarray, Qobj or some other depending on oper_dtype
        self._drift_dyn_gen = None
        self._ctrl_dyn_gen = None
        self._phased_ctrl_dyn_gen = None
        self._dyn_gen_phase = None
        self._initial = None
        self._target = None
        self._onto_evo_target = None
        self._dyn_gen = None
        self._phased_dyn_gen = None
        self._prop = None
        self._prop_grad = None
        self._fwd_evo = None
        self._onwd_evo = None
        self._onto_evo = None
        # The _qobj attribs are Qobj representations of the equivalent
        # internal attribute. They are only set when the extenal accessors
        # are used
        self._onto_evo_target_qobj = None
        self._dyn_gen_qobj = None
        self._prop_qobj = None
        self._prop_grad_qobj = None
        self._fwd_evo_qobj = None
        self._onwd_evo_qobj = None
        self._onto_evo_qobj = None
        # Atrributes used in diagonalisation
        # again in internal operator data type (see above)
        self._decomp_curr = None
        self._prop_eigen = None
        self._dyn_gen_eigenvectors = None
        self._dyn_gen_eigenvectors_adj = None
        self._dyn_gen_factormatrix = None
        self.fact_mat_round_prec = 1e-10

        # Debug and information attribs
        self.stats = None
        self.id_text = 'DYN_BASE'
        self.def_amps_fname = "ctrl_amps.txt"
        self.log_level = self.config.log_level
        # Internal flags
        self._dyn_gen_mapped = False
        self._timeslots_initialized = False
        self._evo_initialized = False
        self._ctrls_initialized = False
        self._ctrls_type_checked = False
        self._drift_type_checked = False
        self._initial_type_checked = False
        self._target_type_checked = False
        # Unitary checking
        self.unitarity_check_level = 0
        self.unitarity_tol = 1e-10
        # Data dumping
        self.dump = None
        self.dump_to_file = False

        self.apply_params()

        # Create the computing objects
        self._create_computers()

        self.clear()

    def apply_params(self, params=None):
        """
        Set object attributes based on the dictionary (if any) passed in the
        instantiation, or passed as a parameter
        This is called during the instantiation automatically.
        The key value pairs are the attribute name and value
        Note: attributes are created if they do not exist already,
        and are overwritten if they do.
        """
        if not params:
            params = self.params

        if isinstance(params, dict):
            self.params = params
            for key in params:
                setattr(self, key, params[key])

    @property
    def log_level(self):
        return logger.level

    @log_level.setter
    def log_level(self, lvl):
        """
        Set the log_level attribute and set the level of the logger
        that is call logger.setLevel(lvl)
        """
        logger.setLevel(lvl)

    @property
    def dumping(self):
        """
        The level of data dumping that will occur during the time evolution
        calculation.
         - NONE : No processing data dumped (Default)
         - SUMMARY : A summary of each time evolution will be recorded
         - FULL : All operators used or created in the calculation dumped
         - CUSTOM : Some customised level of dumping
        When first set to CUSTOM this is equivalent to SUMMARY. It is then up
        to the user to specify which operators are dumped
        WARNING: FULL could consume a lot of memory!
        """
        if self.dump is None:
            lvl = 'NONE'
        else:
            lvl = self.dump.level

        return lvl

    @dumping.setter
    def dumping(self, value):
        if value is None:
            self.dump = None
        else:
            if not _is_string(value):
                raise TypeError("Value must be string value")
            lvl = value.upper()
            if lvl == 'NONE':
                self.dump = None
            else:
                if not isinstance(self.dump, qtrldump.DynamicsDump):
                    self.dump = qtrldump.DynamicsDump(self, level=lvl)
                else:
                    self.dump.level = lvl

    @property
    def dump_dir(self):
        if self.dump:
            return self.dump.dump_dir
        else:
            return None

    @dump_dir.setter
    def dump_dir(self, value):
        if not self.dump:
            self.dumping = 'SUMMARY'
        self.dump.dump_dir = value

    def _create_computers(self):
        """
        Create the default timeslot, fidelity and propagator computers
        """
        # The time slot computer. By default it is set to UpdateAll
        # can be set to DynUpdate in the configuration
        # (see class file for details)
        if self.config.tslot_type == 'DYNAMIC':
            self.tslot_computer = tslotcomp.TSlotCompDynUpdate(self)
        else:
            self.tslot_computer = tslotcomp.TSlotCompUpdateAll(self)

        self.prop_computer = propcomp.PropCompFrechet(self)
        self.fid_computer = fidcomp.FidCompTraceDiff(self)

    def clear(self):
        self.ctrl_amps = None
        self.evo_current = False
        if self.fid_computer is not None:
            self.fid_computer.clear()

    @property
    def num_tslots(self):
        if not self._timeslots_initialized:
            self.init_timeslots()
        return self._num_tslots

    @num_tslots.setter
    def num_tslots(self, value):
        self._num_tslots = value
        if self._timeslots_initialized:
            self._tau = None
            self.init_timeslots()

    @property
    def evo_time(self):
        if not self._timeslots_initialized:
            self.init_timeslots()
        return self._evo_time

    @evo_time.setter
    def evo_time(self, value):
        self._evo_time = value
        if self._timeslots_initialized:
            self._tau = None
            self.init_timeslots()

    @property
    def tau(self):
        if not self._timeslots_initialized:
            self.init_timeslots()
        return self._tau

    @tau.setter
    def tau(self, value):
        self._tau = value
        self.init_timeslots()

    def init_timeslots(self):
        """
        Generate the timeslot duration array 'tau' based on the evo_time
        and num_tslots attributes, unless the tau attribute is already set
        in which case this step in ignored
        Generate the cumulative time array 'time' based on the tau values
        """
        # set the time intervals to be equal timeslices of the total if
        # the have not been set already (as part of user config)
        if self._num_tslots is None:
            self._num_tslots = DEF_NUM_TSLOTS
        if self._evo_time is None:
            self._evo_time = DEF_EVO_TIME

        if self._tau is None:
            self._tau = np.ones(self._num_tslots, dtype='f') * \
                self._evo_time/self._num_tslots
        else:
            self._num_tslots = len(self._tau)
            self._evo_time = np.sum(self._tau)

        self.time = np.zeros(self._num_tslots+1, dtype=float)
        # set the cumulative time by summing the time intervals
        for t in range(self._num_tslots):
            self.time[t+1] = self.time[t] + self._tau[t]

        self._timeslots_initialized = True

    def _set_memory_optimizations(self):
        """
        Set various memory optimisation attributes based on the
        memory_optimization attribute
        If they have been set already, e.g. in apply_params
        then they will not be overidden here
        """
        logger.info("Setting memory optimisations for level {}".format(
                    self.memory_optimization))

        if self.oper_dtype is None:
            self._choose_oper_dtype()
            logger.info("Internal operator data type choosen to be {}".format(
                            self.oper_dtype))
        else:
            logger.info("Using operator data type {}".format(
                            self.oper_dtype))

        if self.cache_phased_dyn_gen is None:
            if self.memory_optimization > 0:
                self.cache_phased_dyn_gen = False
            else:
                self.cache_phased_dyn_gen = True
        logger.info("phased dynamics generator caching {}".format(
                            self.cache_phased_dyn_gen))

        if self.cache_prop_grad is None:
            if self.memory_optimization > 0:
                self.cache_prop_grad = False
            else:
                self.cache_prop_grad = True
        logger.info("propagator gradient caching {}".format(
                            self.cache_prop_grad))

        if self.cache_dyn_gen_eigenvectors_adj is None:
            if self.memory_optimization > 0:
                self.cache_dyn_gen_eigenvectors_adj = False
            else:
                self.cache_dyn_gen_eigenvectors_adj = True
        logger.info("eigenvector adjoint caching {}".format(
                            self.cache_dyn_gen_eigenvectors_adj))

        if self.sparse_eigen_decomp is None:
            if self.memory_optimization > 1:
                self.sparse_eigen_decomp = True
            else:
                self.sparse_eigen_decomp = False
        logger.info("use sparse eigen decomp {}".format(
                            self.sparse_eigen_decomp))

    def _choose_oper_dtype(self):
        """
        Attempt select most efficient internal operator data type
        """

        if self.memory_optimization > 0:
            self.oper_dtype = Qobj
        else:
            # Method taken from Qobj.expm()
            # if method is not explicitly given, try to make a good choice
            # between sparse and dense solvers by considering the size of the
            # system and the number of non-zero elements.
            if isinstance(self.drift_dyn_gen, (list, tuple)):
                dg = self.drift_dyn_gen[0]
            else:
                dg = self.drift_dyn_gen
            if isinstance(self.ctrl_dyn_gen[0], (list, tuple)):
                ctrls = self.ctrl_dyn_gen[0]
            else:
                ctrls = self.ctrl_dyn_gen
            for c in ctrls:
               dg = dg + c

            N = dg.data.shape[0]
            n = dg.data.nnz

            if N ** 2 < 100 * n:
                # large number of nonzero elements, revert to dense solver
                self.oper_dtype = np.ndarray
            elif N > 400:
                # large system, and quite sparse -> qutips sparse method
                self.oper_dtype = Qobj
            else:
                # small system, but quite sparse -> qutips sparse/dense method
                self.oper_dtype = np.ndarray

        return self.oper_dtype

    def _init_evo(self):
        """
        Create the container lists / arrays for the:
        dynamics generations, propagators, and evolutions etc
        Set the time slices and cumulative time
        """
        # check evolution operators
        ne = _check_ensemble_size(self.ensemble_size)
        self.ensemble_size = ne

        if not self._initial_type_checked:
            _check_initial(self.initial)
            self._initial_type_checked = True
        if not self._target_type_checked:
            _check_target(self.target)
            self._target_type_checked = True
        if not self._drift_type_checked:
            _check_drift(self.drift_dyn_gen)
            self._drift_type_checked = True
        if not self._ctrls_type_checked:
            _check_ctrls(self.ctrl_dyn_gen)
            self._ctrls_type_checked = True

        if not isinstance(self.initial, Qobj):
            raise TypeError("initial must be a Qobj")

        if not isinstance(self.target, Qobj):
            raise TypeError("target must be a Qobj")

        self.refresh_drift_attribs()
        self._set_memory_optimizations()

        n_ts = self.num_tslots
        n_ctrls = self.num_ctrls

        self._initial = []
        self._target = []
        self._drift_dyn_gen = []
        self._ctrl_dyn_gen = []

        # check ensemble set up
        self._e_shares_initial = True
        self._e_shares_drift = True
        self._e_shares_ctrls = True
        self._e_shares_target = True
        if isinstance(self.initial, (list, tuple)):
            if len(self.initial) == ne:
                self._e_shares_initial = False
            else:
                raise ValueError("List len {} of initial evo does not match "
                            "ensemble_size {}".format(len(self.initial), ne))
        if isinstance(self.target, list):
            if len(self.target) == ne:
                self._e_shares_target = False
            else:
                raise ValueError("List len {} of target evo does not match "
                            "ensemble_size {}".format(len(self.target), ne))
        # These lists don't have to be the same size, as drift could be
        # time dependent list
        if (isinstance(self.drift_dyn_gen, (list, tuple))
                and len(self.drift_dyn_gen)) == ne:
            self._e_shares_drift = False

        # self.ctrl_dyn_gen will always be a list
        # there could confusion here if n_ctrls==ne and there are
        # time dependent controls
        if (isinstance(self.ctrl_dyn_gen[0], (list, tuple))
                and len(self.ctrl_dyn_gen) == ne):
            self._e_shares_ctrls = False


        if self._e_shares_initial:
            self._sys_dims = [self.initial.dims for e in range(ne)]
            self._sys_shape = [self.initial.shape for e in range(ne)]
        else:
            self._sys_dims = [self.initial[e].dims for e in range(ne)]
            self._sys_shape = [self.initial[e].shape for e in range(ne)]
        #TODO: Check matching evo and operator types

        # Set up the dynamics operators for a given ensembles
        # Note there is some attempt here to avoid making copies of objects
        if self.oper_dtype == Qobj:
            if self._e_shares_initial:
                self._initial = [self.initial for e in range(ne)]
            else:
                self._initial = self.initial

            if self._e_shares_target:
                self._target = [self.target for e in range(ne)]
            else:
                self._target = self.target

            if self._e_shares_drift:
                self._drift_dyn_gen = [self.drift_dyn_gen for e in range(ne)]
            else:
                self._drift_dyn_gen = self.drift_dyn_gen

            if self._e_shares_ctrls:
                self._ctrl_dyn_gen = [self.ctrl_dyn_gen for e in range(ne)]
            else:
                self._ctrl_dyn_gen = self.ctrl_dyn_gen

        elif self.oper_dtype == np.ndarray:
            if self._e_shares_initial:
                dense = self.initial.full()
                self._initial = [dense for e in range(ne)]
            else:
                self._initial = [evo.full() for evo in self.initial]

            if self._e_shares_target:
                dense = self.target.full()
                self._target = [dense for e in range(ne)]
            else:
                self._target = self.target

            if self._e_shares_drift:
                dense = _create_nested_dense(self.drift_dyn_gen)
                self._drift_dyn_gen = [dense for e in range(ne)]
            else:
                self._drift_dyn_gen = _create_nested_dense(self.drift_dyn_gen)

            if self._e_shares_ctrls:
                dense = _create_nested_dense(self.ctrl_dyn_gen)
                self._ctrl_dyn_gen = [dense for e in range(ne)]
            else:
                self._ctrl_dyn_gen = _create_nested_dense(self.drift_dyn_gen)

        else:
            logger.warn("Unknown option '{}' for oper_dtype. "
                "Assuming that internal drift, ctrls, initial and target "
                "have been set correctly".format(self.oper_dtype))

        #TODO: check for td drift and controls
        # For now just set td to false

        self._time_depend_drift = [False for e in range(ne)]
        self._time_depend_ctrl_dyn_gen = [False for e in range(ne)]


        if self.cache_phased_dyn_gen and not self.dyn_gen_phase is None:

            self._phased_ctrl_dyn_gen = [[self._apply_phase(ctrl)
                                          for ctrl in self._ctrl_dyn_gen[e]]
                                         for e in range(ne)]

        self._dyn_gen = [[object for k in range(n_ts)]
                         for e in range(ne)]
        if self.cache_phased_dyn_gen:
            self._phased_dyn_gen = [[object for k in range(n_ts)]
                                    for e in range(ne)]
        self._prop = [[object for k in range(n_ts)]
                      for e in range(ne)]
        if self.prop_computer.grad_exact and self.cache_prop_grad:
            self._prop_grad = [[[object for j in range(n_ctrls)]
                                for k in range(n_ts)]
                               for e in range(ne)]
        # Time evolution operator (forward propagation)
        self._fwd_evo = [[object for k in range(n_ts+1)] for e in range(ne)]
        for e in range(ne):
            self._fwd_evo[e][0] = self._initial[e]
        if self.fid_computer.uses_onwd_evo:
            # Time evolution operator (onward propagation)
            self._onwd_evo = [[object for k in range(n_ts)] for e in range(ne)]
        if self.fid_computer.uses_onto_evo:
            # Onward propagation overlap with inverse target
            self._onto_evo = [[object for k in range(n_ts+1)]
                              for e in range(ne)]
            self._onto_evo_target = [object for e in range(ne)]
            for e in range(ne):
                self._onto_evo[e][n_ts] = self._get_onto_evo_target(e)

        if isinstance(self.prop_computer, propcomp.PropCompDiag):
            self._create_decomp_lists()

        if (self.log_level <= logging.DEBUG
            and isinstance(self, DynamicsUnitary)):
                self.unitarity_check_level = 1

        if self.dump_to_file:
            if self.dump is None:
                self.dumping = 'SUMMARY'
            self.dump.write_to_file = True
            self.dump.create_dump_dir()
            logger.info("Dynamics dump will be written to:\n{}".format(
                            self.dump.dump_dir))

        self._evo_initialized = True

    def _create_decomp_lists(self):
        """
        Create lists that will hold the eigen decomposition
        used in calculating propagators and gradients
        Note: used with PropCompDiag propagator calcs
        """
        n_ts = self.num_tslots
        ne = self.ensemble_size
        self._decomp_curr = [[False for k in range(n_ts)] for e in range(ne)]
        self._prop_eigen = [[object for k in range(n_ts)] for e in range(ne)]
        self._dyn_gen_eigenvectors = [[object for k in range(n_ts)]
                                      for e in range(ne)]
        if self.cache_dyn_gen_eigenvectors_adj:
            self._dyn_gen_eigenvectors_adj = [[object for k in range(n_ts)]
                                              for e in range(ne)]
        self._dyn_gen_factormatrix = [[object for k in range(n_ts)]
                                      for e in range(ne)]

    def initialize_controls(self, amps, init_tslots=True, init_evo=True):
        """
        Set the initial control amplitudes and time slices
        Note this must be called after the configuration is complete
        before any dynamics can be calculated
        """

        if not self._timeslots_initialized:
            init_tslots = True
        if not self._evo_initialized:
            init_evo = True
        if not isinstance(self.prop_computer, propcomp.PropagatorComputer):
            raise errors.UsageError(
                "No prop_computer (propagator computer) "
                "set. A default should be assigned by the Dynamics subclass")

        if not isinstance(self.tslot_computer, tslotcomp.TimeslotComputer):
            raise errors.UsageError(
                "No tslot_computer (Timeslot computer)"
                " set. A default should be assigned by the Dynamics class")

        if not isinstance(self.fid_computer, fidcomp.FidelityComputer):
            raise errors.UsageError(
                "No fid_computer (Fidelity computer)"
                " set. A default should be assigned by the Dynamics subclass")

        self.ctrl_amps = None
        if init_tslots:
            self.init_timeslots()
        if init_evo:
            self._init_evo()
        self.tslot_computer.init_comp()
        self.fid_computer.init_comp()
        self._ctrls_initialized = True
        self.update_ctrl_amps(amps)

    def check_ctrls_initialized(self):
        if not self._ctrls_initialized:
            raise errors.UsageError(
                "Controls not initialised. "
                "Ensure Dynamics.initialize_controls has been "
                "executed with the initial control amplitudes.")

    def get_amp_times(self):
        return self.time[:self.num_tslots]

    def save_amps(self, file_name=None, times=None, amps=None, verbose=False):
        """
        Save a file with the current control amplitudes in each timeslot
        The first column in the file will be the start time of the slot

        Parameters
        ----------
        file_name : string
            Name of the file
            If None given the def_amps_fname attribuite will be used

        times : List type (or string)
            List / array of the start times for each slot
            If None given this will be retrieved through get_amp_times()
            If 'exclude' then times will not be saved in the file, just
            the amplitudes

        amps : Array[num_tslots, num_ctrls]
            Amplitudes to be saved
            If None given the ctrl_amps attribute will be used

        verbose : Boolean
            If True then an info message will be logged
        """
        self.check_ctrls_initialized()

        inctimes = True
        if file_name is None:
            file_name = self.def_amps_fname
        if amps is None:
            amps = self.ctrl_amps
        if times is None:
            times = self.get_amp_times()
        else:
            if _is_string(times):
                if times.lower() == 'exclude':
                    inctimes = False
                else:
                    logger.warn("Unknown option for times '{}' "
                                "when saving amplitudes".format(times))
                    times = self.get_amp_times()

        try:
            if inctimes:
                shp = amps.shape
                data = np.empty([shp[0], shp[1] + 1], dtype=float)
                data[:, 0] = times
                data[:, 1:] = amps
            else:
                data = amps

            np.savetxt(file_name, data, delimiter='\t', fmt='%14.6g')

            if verbose:
                logger.info("Amplitudes saved to file: " + file_name)
        except Exception as e:
            logger.error("Failed to save amplitudes due to underling "
                         "error: {}".format(e))

    def update_ctrl_amps(self, new_amps):
        """
        Determine if any amplitudes have changed. If so, then mark the
        timeslots as needing recalculation
        The actual work is completed by the compare_amps method of the
        timeslot computer
        """

        if self.log_level <= logging.DEBUG_INTENSE:
            logger.log(logging.DEBUG_INTENSE, "Updating amplitudes...\n"
                       "Current control amplitudes:\n" + str(self.ctrl_amps) +
                       "\n(potenially) new amplitudes:\n" + str(new_amps))

        self.tslot_computer.compare_amps(new_amps)

    def flag_system_changed(self):
        """
        Flag evolution, fidelity and gradients as needing recalculation
        """
        self.evo_current = False
        self.fid_computer.flag_system_changed()

    @property
    def time_depend_drift(self):
        if not self._evo_initialized:
            raise RuntimeError("Evolution must be initialised before "
                               "time dependence can be determined")
        if self.ensemble_size == 1:
            return self._time_depend_drift[0]
        else:
            return self._time_depend_drift

    @property
    def time_depend_ctrl_dyn_gen(self):
        if not self._evo_initialized:
            raise RuntimeError("Evolution must be initialised before "
                               "time dependence can be determined")
        if self.ensemble_size == 1:
            return self._time_depend_ctrl_dyn_gen[0]
        else:
            return self._time_depend_ctrl_dyn_gen

    def get_drift_dim(self, e=0):
        """
        Returns the size of the matrix that defines the drift dynamics
        that is assuming the drift is NxN, then this returns N
        """
        if self._dyn_shape is None:
            self.refresh_drift_attribs()
        return self._dyn_shape[e][0]

    def refresh_drift_attribs(self):
        """Reset the dyn_shape, dyn_dims and time_depend_drift attribs"""
        # This may need to work before _evo_initialized
        # TODO: Move into _init_evo
        if isinstance(self.drift_dyn_gen, (list, tuple)):
            self._dyn_shape = [self.drift_dyn_gen[e].shape
                               for e in range(self.ensemble_size)]
            self._dyn_dims = [self.drift_dyn_gen[e].dims
                               for e in range(self.ensemble_size)]
        else:
            self._dyn_shape = [self.drift_dyn_gen.shape
                               for e in range(self.ensemble_size)]
            self._dyn_dims = [self.drift_dyn_gen.dims
                               for e in range(self.ensemble_size)]

        #FIXME: This is a fudge for now
#        self.dyn_shape = d0.shape[0]
#        self.dyn_dims = d0.dims[0]

    def get_num_ctrls(self):
        """
        calculate the of controls from the length of the control list
        sets the num_ctrls property, which can be used alternatively
        subsequently
        """
        _func_deprecation("'get_num_ctrls' has been replaced by "
                         "'num_ctrls' property")
        return self.num_ctrls

    def _get_num_ctrls(self):
        #FIXME: Will not work with td ctrls
        if not self._ctrls_type_checked:
            _check_ctrls(self.ctrl_dyn_gen)
            self._ctrls_type_checked = True

        if isinstance(self.ctrl_dyn_gen[0], (list, tuple)):
            self._num_ctrls = len(self.ctrl_dyn_gen[0])
        else:
            self._num_ctrls = len(self.ctrl_dyn_gen)

        return self._num_ctrls

    @property
    def num_ctrls(self):
        """
        calculate the of controls from the length of the control list
        sets the num_ctrls property, which can be used alternatively
        subsequently
        """
        if self._num_ctrls is None:
            self._num_ctrls = self._get_num_ctrls()
        return self._num_ctrls

    @property
    def onto_evo_target(self):
        raise NotImplementedError("Not implemented this yet")
        if self._onto_evo_target is None:
            self._get_onto_evo_target()

        if self._onto_evo_target_qobj is None:
            if isinstance(self._onto_evo_target, Qobj):
                self._onto_evo_target_qobj = self._onto_evo_target
            else:
                rev_dims = [self._sys_dims[1], self._sys_dims[0]]
                self._onto_evo_target_qobj = Qobj(self._onto_evo_target,
                                                  dims=rev_dims)

        return self._onto_evo_target_qobj

    def get_owd_evo_target(self):
        _func_deprecation("'get_owd_evo_target' has been replaced by "
                         "'onto_evo_target' property")
        return self.onto_evo_target

    def _get_onto_evo_target(self, e=0):
        """
        Get the inverse of the target.
        Used for calculating the 'onto target' evolution
        This is actually only relevant for unitary dynamics where
        the target.dag() is what is required
        However, for completeness, in general the inverse of the target
        operator is is required
        For state-to-state, the bra corresponding to the is required ket
        """
        #self._onto_evo_target = [object for e in self.ensemble_size]

        if self.target.shape[0] == self.target.shape[1]:
            #Target is operator
            targ = la.inv(self.target.full())
            if self.oper_dtype == Qobj:
                self._onto_evo_target[e] = Qobj(targ)
            elif self.oper_dtype == np.ndarray:
                self._onto_evo_target[e] = targ
            else:
                raise TypeError("Invalid oper_dtype "
                                "{}".format(type(self.oper_dtype)))
        else:
            if self.oper_dtype == Qobj:
                self._onto_evo_target[e] = self.target.dag()
            elif self.oper_dtype == np.ndarray:
                self._onto_evo_target[e] = self.target.dag().full()
            else:
                raise TypeError("Invalid oper_dtype "
                                "{}".format(type(self.oper_dtype)))

        return self._onto_evo_target[e]

    def combine_dyn_gen(self, k):
        """
        Computes the dynamics generator for a given timeslot
        The is the combined Hamiltion for unitary systems
        """
        _func_deprecation("'combine_dyn_gen' has been replaced by "
                        "'_combine_dyn_gen'")
        self._combine_dyn_gen(k)
        return self._dyn_gen(k)

    def _combine_dyn_gen(self, e, k):
        """
        Computes the dynamics generator for a given timeslot
        The is the combined Hamiltion for unitary systems
        Also applies the phase (if any required by the propagation)
        """
        dg = self._drift_dyn_gen[e]
        for j in range(self._num_ctrls):
            dg = dg + self.ctrl_amps[k, j]*self._ctrl_dyn_gen[e][j]

        self._dyn_gen[e][k] = dg
        if self.cache_phased_dyn_gen:
            self._phased_dyn_gen[e][k] = self._apply_phase(dg)

    @property
    def dyn_gen_phase(self):
        """
        Some preop that is applied to the dyn_gen before expontiating to
        get the propagator
        """
        return self._dyn_gen_phase

    def _apply_phase(self, dg):
        """
        Apply some phase factor or operator
        """
        if self.dyn_gen_phase is None:
            phased_dg = dg
        else:
            if hasattr(self.dyn_gen_phase, 'dot'):
                phased_dg = self.dyn_gen_phase.dot(dg)
            else:
                phased_dg = self.dyn_gen_phase*dg
        return phased_dg

    def get_dyn_gen(self, e, k):
        """
        Get the combined dynamics generator for the timeslot
        Not implemented in the base class. Choose a subclass
        """
        _func_deprecation("'get_dyn_gen' has been replaced by "
                        "'_get_phased_dyn_gen'")
        return self._get_phased_dyn_gen(e, k)

    def _get_phased_dyn_gen(self, e, k):
        if self.dyn_gen_phase is None:
            return self._dyn_gen[e][k]
        else:
            if self._phased_dyn_gen is None:
                return self._apply_phase(self._dyn_gen[e][k])
            else:
                return self._phased_dyn_gen[e][k]

    def get_ctrl_dyn_gen(self, e, j):
        """
        Get the dynamics generator for the control
        Not implemented in the base class. Choose a subclass
        """
        _func_deprecation("'get_ctrl_dyn_gen' has been replaced by "
                        "'_get_phased_ctrl_dyn_gen'")
        return self._get_phased_ctrl_dyn_gen(e, 0, j)

    def _get_phased_ctrl_dyn_gen(self, e, k, j):
        if self._phased_ctrl_dyn_gen is not None:
            return self._phased_ctrl_dyn_gen[e][j]
        else:
            return self._apply_phase(self._ctrl_dyn_gen[e][j])

    @property
    def dyn_gen(self):
        """
        List of combined dynamics generators (Qobj) for each timeslot
        """
        if self._dyn_gen is not None:
            if self._dyn_gen_qobj is None:
                if self.oper_dtype == Qobj:
                    self._dyn_gen_qobj = self._dyn_gen
                else:
                    self._dyn_gen_qobj = [[Qobj(dg, dims=self.dyn_dims)
                                            for dg in self._dyn_gen[e]]
                                          for e in self.ensemble_size]
        return self._dyn_gen_qobj

    @property
    def prop(self):
        """
        List of propagators (Qobj) for each timeslot
        """
        raise NotImplementedError("ensemble")
        if self._prop is not None:
            if self._prop_qobj is None:
                if self.oper_dtype == Qobj:
                    self._prop_qobj = self._prop
                else:
                    self._prop_qobj = [Qobj(dg, dims=self.dyn_dims)
                                            for dg in self._prop]
        return self._prop_qobj

    @property
    def prop_grad(self):
        """
        Array of propagator gradients (Qobj) for each timeslot, control
        """
        raise NotImplementedError("ensemble")
        if self._prop_grad is not None:
            if self._prop_grad_qobj is None:
                if self.oper_dtype == Qobj:
                    self._prop_grad_qobj = self._prop_grad
                else:
                    self._prop_grad_qobj = np.empty(
                                    [self.num_tslots, self.num_ctrls],
                                    dtype=object)
                    for k in range(self.num_tslots):
                        for j in range(self.num_ctrls):
                            self._prop_grad_qobj[k, j] = Qobj(
                                                    self._prop_grad[k, j],
                                                    dims=self.dyn_dims)
        return self._prop_grad_qobj

    def _get_prop_grad(self, e, k, j):
        if self.cache_prop_grad:
            prop_grad = self._prop_grad[e][k][j]
        else:
            prop_grad = self.prop_computer._compute_prop_grad(e, k, j,
                                                       compute_prop = False)
        return prop_grad

    @property
    def evo_init2t(self):
        _attrib_deprecation(
            "'evo_init2t' has been replaced by '_fwd_evo'")
        return self._fwd_evo

    @property
    def fwd_evo(self):
        """
        List of evolution operators (Qobj) from the initial to the given
        timeslot
        """
        #raise NotImplementedError("ensemble")
        if self._fwd_evo is not None:
            if self._fwd_evo_qobj is None:
                if self.oper_dtype == Qobj:
                    self._fwd_evo_qobj = self._fwd_evo
                else:
                    self._fwd_evo_qobj = [[Qobj(self._fwd_evo[e][k],
                                                       dims=self._sys_dims[e])
                                           for k in range(self.num_tslots+1)]
                                          for e in range(self.ensemble_size)]
#                    self._fwd_evo_qobj = [self.initial]
#                    for k in range(1, self.num_tslots+1):
#                        self._fwd_evo_qobj.append(Qobj(self._fwd_evo[k],
#                                                       dims=self._sys_dims))
        return self._fwd_evo_qobj

    def _get_full_evo(self):
        return self._fwd_evo[self._num_tslots]

    @property
    def full_evo(self):
        """Full evolution - time evolution at final time slot"""
        return [self.fwd_evo[e][self.num_tslots]
                for e in range(self.ensemble_size)]

    @property
    def evo_t2end(self):
        _attrib_deprecation(
            "'evo_t2end' has been replaced by '_onwd_evo'")
        return self._onwd_evo

    @property
    def onwd_evo(self):
        """
        List of evolution operators (Qobj) from the initial to the given
        timeslot
        """
        raise NotImplementedError("ensemble")
        if self._onwd_evo is not None:
            if self._onwd_evo_qobj is None:
                if self.oper_dtype == Qobj:
                    self._onwd_evo_qobj = self._fwd_evo
                else:
                    self._onwd_evo_qobj = [Qobj(dg, dims=self._sys_dims)
                                            for dg in self._onwd_evo]
        return self._onwd_evo_qobj

    @property
    def evo_t2targ(self):
        _attrib_deprecation(
            "'evo_t2targ' has been replaced by '_onto_evo'")
        return self._onto_evo

    @property
    def onto_evo(self):
        """
        List of evolution operators (Qobj) from the initial to the given
        timeslot
        """
        raise NotImplementedError("ensemble")
        if self._onto_evo is not None:
            if self._onto_evo_qobj is None:
                if self.oper_dtype == Qobj:
                    self._onto_evo_qobj = self._onto_evo
                else:
                    self._onto_evo_qobj = []
                    for k in range(0, self.num_tslots):
                        self._onto_evo_qobj.append(Qobj(self._onto_evo[k],
                                                       dims=self._sys_dims))
                    self._onto_evo_qobj.append(self.onto_evo_target)

        return self._onto_evo_qobj

    def compute_evolution(self):
        """
        Recalculate the time evolution operators
        Dynamics generators (e.g. Hamiltonian) and
        prop (propagators) are calculated as necessary
        Actual work is completed by the recompute_evolution method
        of the timeslot computer
        """

        # Check if values are already current, otherwise calculate all values
        if not self.evo_current:
            if self.log_level <= logging.DEBUG_VERBOSE:
                logger.log(logging.DEBUG_VERBOSE, "Computing evolution")
            self.tslot_computer.recompute_evolution()
            self.evo_current = True
            return True
        else:
            return False

    def _ensure_decomp_curr(self, e, k):
        """
        Checks to see if the diagonalisation has been completed since
        the last update of the dynamics generators
        (after the amplitude update)
        If not then the diagonlisation is completed
        """
        if self._decomp_curr is None:
            raise errors.UsageError("Decomp lists have not been created")
        if not self._decomp_curr[e][k]:
            self._spectral_decomp(e, k)

    def _spectral_decomp(self, e, k):
        """
        Calculate the diagonalization of the dynamics generator
        generating lists of eigenvectors, propagators in the diagonalised
        basis, and the 'factormatrix' used in calculating the propagator
        gradient
        Not implemented in this base class, because the method is specific
        to the matrix type
        """
        raise errors.UsageError("Decomposition cannot be completed by "
                                "this class. Try a(nother) subclass")

    def _is_unitary(self, A):
        """
        Checks whether operator A is unitary
        A can be either Qobj or ndarray
        """
        if isinstance(A, Qobj):
            unitary = np.allclose(np.eye(A.shape[0]), A*A.dag().full(),
                        atol=self.unitarity_tol)
        else:
            unitary = np.allclose(np.eye(len(A)), A.dot(A.T.conj()),
                        atol=self.unitarity_tol)

        return unitary

    def _calc_unitary_err(self, A):
        if isinstance(A, Qobj):
            err = np.sum(abs(np.eye(A.shape[0]) - A*A.dag().full()))
        else:
            err = np.sum(abs(np.eye(len(A)) - A.dot(A.T.conj())))

        return err

    def unitarity_check(self):
        """
        Checks whether all propagators are unitary
        """
        for e in range(self.ensemble_size):
            for k in range(self.num_tslots):
                if not self._is_unitary(self._prop[e][k]):
                    logger.warning(
                        "Progator of system {} timeslot {} "
                        "is not unitary".format(e, k))


class DynamicsGenMat(Dynamics):
    """
    This sub class can be used for any system where no additional
    operator is applied to the dynamics generator before calculating
    the propagator, e.g. classical dynamics, Lindbladian
    """
    def reset(self):
        Dynamics.reset(self)
        self.id_text = 'GEN_MAT'
        self.apply_params()

class DynamicsUnitary(Dynamics):
    """
    This is the subclass to use for systems with dynamics described by
    unitary matrices. E.g. closed systems with Hermitian Hamiltonians
    Note a matrix diagonalisation is used to compute the exponent
    The eigen decomposition is also used to calculate the propagator gradient.
    The method is taken from DYNAMO (see file header)

    Attributes
    ----------
    drift_ham : Qobj
        This is the drift Hamiltonian for unitary dynamics
        It is mapped to drift_dyn_gen during initialize_controls

    ctrl_ham : List of Qobj
        These are the control Hamiltonians for unitary dynamics
        It is mapped to ctrl_dyn_gen during initialize_controls

    H : List of Qobj
        The combined drift and control Hamiltonians for each timeslot
        These are the dynamics generators for unitary dynamics.
        It is mapped to dyn_gen during initialize_controls
    """

    def reset(self):
        Dynamics.reset(self)
        self.id_text = 'UNIT'
        self.drift_ham = None
        self.ctrl_ham = None
        self.H = None
        self._dyn_gen_phase = -1j
        self.apply_params()

    def _create_computers(self):
        """
        Create the default timeslot, fidelity and propagator computers
        """
        # The time slot computer. By default it is set to _UpdateAll
        # can be set to _DynUpdate in the configuration
        # (see class file for details)
        if self.config.tslot_type == 'DYNAMIC':
            self.tslot_computer = tslotcomp.TSlotCompDynUpdate(self)
        else:
            self.tslot_computer = tslotcomp.TSlotCompUpdateAll(self)

        # set the default fidelity computer
        self.fid_computer = fidcomp.FidCompUnitary(self)
        # set the default propagator computer
        self.prop_computer = propcomp.PropCompDiag(self)

    def initialize_controls(self, amplitudes, init_tslots=True):
        # Either the _dyn_gen or _ham names can be used
        # This assumes that one or other has been set in the configuration

        self._map_dyn_gen_to_ham()
        Dynamics.initialize_controls(self, amplitudes, init_tslots=init_tslots)
        #self.H = self._dyn_gen

    def _map_dyn_gen_to_ham(self):
        if self.drift_dyn_gen is None:
            self.drift_dyn_gen = self.drift_ham
        else:
            self.drift_ham = self.drift_dyn_gen

        if self.ctrl_dyn_gen is None:
            self.ctrl_dyn_gen = self.ctrl_ham
        else:
            self.ctrl_ham = self.ctrl_dyn_gen

        self._dyn_gen_mapped = True

    @property
    def num_ctrls(self):
        if not self._dyn_gen_mapped:
            self._map_dyn_gen_to_ham()
        if self._num_ctrls is None:
            self._num_ctrls = self._get_num_ctrls()
        return self._num_ctrls

    def _get_onto_evo_target(self, e):
        """
        Get the adjoint of the target.
        Used for calculating the 'backward' evolution
        """
        if self.oper_dtype == Qobj:
            self._onto_evo_target[e] = self._target[e].dag()
        else:
            self._onto_evo_target[e] = self._target[e].T.conj()
        return self._onto_evo_target[e]

    def _spectral_decomp(self, e, k):
        """
        Calculates the diagonalization of the dynamics generator
        generating lists of eigenvectors, propagators in the diagonalised
        basis, and the 'factormatrix' used in calculating the propagator
        gradient
        """

        if self.oper_dtype == Qobj:
            H = self._dyn_gen[e][k]
            # Returns eigenvalues as array (row)
            # and eigenvectors as rows of an array
            eig_val, eig_vec = sp_eigs(H.data, H.isherm,
                                       sparse=self.sparse_eigen_decomp)
            eig_vec = eig_vec.T

        elif self.oper_dtype == np.ndarray:
            H = self._dyn_gen[e][k]
            # returns row vector of eigenvals, columns with the eigenvecs
            eig_val, eig_vec = np.linalg.eigh(H)
        else:
            raise TypeError

        # assuming H is an nxn matrix, find n
        n = self.get_drift_dim(e)

        # Calculate the propagator in the diagonalised basis
        eig_val_tau = -1j*eig_val*self.tau[k]
        prop_eig = np.exp(eig_val_tau)

        # Generate the factor matrix through the differences
        # between each of the eigenvectors and the exponentiations
        # create nxn matrix where each eigen val is repeated n times
        # down the columns
        o = np.ones([n, n])
        eig_val_cols = eig_val_tau*o
        # calculate all the differences by subtracting it from its transpose
        eig_val_diffs = eig_val_cols - eig_val_cols.T
        # repeat for the propagator
        prop_eig_cols = prop_eig*o
        prop_eig_diffs = prop_eig_cols - prop_eig_cols.T
        # the factor matrix is the elementwise quotient of the
        # differeneces between the exponentiated eigen vals and the
        # differences between the eigen vals
        # need to avoid division by zero that would arise due to denegerate
        # eigenvalues and the diagonals
        degen_mask = np.abs(eig_val_diffs) < self.fact_mat_round_prec
        eig_val_diffs[degen_mask] = 1
        factors = prop_eig_diffs / eig_val_diffs
        # for degenerate eigenvalues the factor is just the exponent
        factors[degen_mask] = prop_eig_cols[degen_mask]

        # Store eigenvectors, propagator and factor matric
        # for use in propagator computations
        self._decomp_curr[e][k] = True
        if isinstance(factors, np.ndarray):
            self._dyn_gen_factormatrix[e][k] = factors
        else:
            self._dyn_gen_factormatrix[e][k] = np.array(factors)

        if self.oper_dtype == Qobj:
            self._prop_eigen[e][k] = Qobj(np.diagflat(prop_eig),
                                                    dims=self._dyn_dims[e])
            self._dyn_gen_eigenvectors[e][k] = Qobj(eig_vec,
                                                dims=self._dyn_dims[e])
            # The _dyn_gen_eigenvectors_adj list is not used in
            # memory optimised modes
            if self._dyn_gen_eigenvectors_adj is not None:
                self._dyn_gen_eigenvectors_adj[e][k] = \
                            self._dyn_gen_eigenvectors[e][k].dag()
        else:
            self._prop_eigen[e][k] = np.diagflat(prop_eig)
            self._dyn_gen_eigenvectors[e][k] = eig_vec
            # The _dyn_gen_eigenvectors_adj list is not used in
            # memory optimised modes
            if self._dyn_gen_eigenvectors_adj is not None:
                self._dyn_gen_eigenvectors_adj[e][k] = \
                            self._dyn_gen_eigenvectors[e][k].conj().T

    def _get_dyn_gen_eigenvectors_adj(self, e, k):
        # The _dyn_gen_eigenvectors_adj list is not used in
        # memory optimised modes
        if self._dyn_gen_eigenvectors_adj is not None:
            return self._dyn_gen_eigenvectors_adj[e][k]
        else:
            if self.oper_dtype == Qobj:
                return self._dyn_gen_eigenvectors[e][k].dag()
            else:
                return self._dyn_gen_eigenvectors[e][k].conj().T

    def check_unitarity(self):
        """
        Checks whether all propagators are unitary
        For propagators found not to be unitary, the potential underlying
        causes are investigated.
        """
        for e in range(self.ensemble_size):
            for k in range(self.num_tslots):
                prop_unit = self._is_unitary(self._prop[e][k])
                if not prop_unit:
                    logger.warning(
                        "Progator of system {} timeslot {} "
                        "is not unitary".format(e, k))
                if not prop_unit or self.unitarity_check_level > 1:
                    # Check Hamiltonian
                    H = self._dyn_gen[e][k]
                    if isinstance(H, Qobj):
                        herm = H.isherm
                    else:
                        diff = np.abs(H.T.conj() - H)
                        herm = False if np.any(diff > settings.atol) else True
                    eigval_unit = self._is_unitary(self._prop_eigen[e][k])
                    eigvec_unit = self._is_unitary(self._dyn_gen_eigenvectors[e][k])
                    if self._dyn_gen_eigenvectors_adj is not None:
                        eigvecadj_unit = self._is_unitary(
                                        self._dyn_gen_eigenvectors_adj[e][k])
                    else:
                        eigvecadj_unit = None
                    msg = ("prop unit: {}; H herm: {}; "
                            "eigval unit: {}; eigvec unit: {}; "
                            "eigvecadj_unit: {}".format(
                            prop_unit, herm, eigval_unit, eigvec_unit,
                                eigvecadj_unit))
                    logger.info(msg)

class DynamicsSymplectic(Dynamics):
    """
    Symplectic systems
    This is the subclass to use for systems where the dynamics is described
    by symplectic matrices, e.g. coupled oscillators, quantum optics

    Attributes
    ----------
    omega : array[drift_dyn_gen.shape]
        matrix used in the calculation of propagators (time evolution)
        with symplectic systems.

    """

    def reset(self):
        Dynamics.reset(self)
        self.id_text = 'SYMPL'
        self._omega = None
        self._omega_qobj = None
        self.grad_exact = True
        self.apply_params()

    def _create_computers(self):
        """
        Create the default timeslot, fidelity and propagator computers
        """
        # The time slot computer. By default it is set to _UpdateAll
        # can be set to _DynUpdate in the configuration
        # (see class file for details)
        if self.config.tslot_type == 'DYNAMIC':
            self.tslot_computer = tslotcomp.TSlotCompDynUpdate(self)
        else:
            self.tslot_computer = tslotcomp.TSlotCompUpdateAll(self)

        self.prop_computer = propcomp.PropCompFrechet(self)
        self.fid_computer = fidcomp.FidCompTraceDiff(self)

    @property
    def omega(self):
        if self._omega is None:
            self._get_omega()
        if self._omega_qobj is None:
            self._omega_qobj = Qobj(self._omega, dims=self.dyn_dims)
        return self._omega_qobj

    def _get_omega(self):
        if self._omega is None:
            n = self.get_drift_dim() // 2
            omg = sympl.calc_omega(n)
            if self.oper_dtype == Qobj:
                self._omega = Qobj(omg, dims=self.dyn_dims)
                self._omega_qobj = self._omega
            elif self.oper_dtype == sp.csr_matrix:
                self._omega = sp.csr_matrix(omg)
            else:
                 self._omega = omg
        return self._omega

    @property
    def dyn_gen_phase(self):
        """
        The prephasing operator for the symplectic group generators
        usually refered to as \Omega
        """
        # Cannot be calculated until the dyn_shape is set
        # that is after the drift Hamitonan has been set.
        if self._dyn_gen_phase is None:
            self._dyn_gen_phase = self._get_omega()

        return self._dyn_gen_phase

    def _apply_phase(self, dg):
        """
        Apply some phase factor or operator
        """
        if self.dyn_gen_phase is None:
            phased_dg = dg
        else:
            if hasattr(self.dyn_gen_phase, 'dot'):
                phased_dg = -dg.dot(self.dyn_gen_phase)
            else:
                phased_dg = -dg*self.dyn_gen_phase
        return phased_dg