# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson,
#                      Neill Lambert, Anubhav Vardhan, Alexander Pitchford.
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
"""
This module provides exact solvers for a system-bath setup using the
hierarchy equations of motion (HEOM).
"""

# Authors: Neill Lambert, Anubhav Vardhan, Alexander Pitchford
# Contact: nwlambert@gmail.com

import types
from functools import partial
import warnings
import timeit
import numpy as np
#from scipy.misc import factorial
import scipy.sparse as sp
import scipy.integrate
from copy import copy
from qutip import Qobj, qeye
from qutip.states import enr_state_dictionaries
from qutip.superoperator import liouvillian, spre, spost
from qutip.cy.spmatfuncs import cy_ode_rhs
from qutip.solver import Options, Result, Stats
from qutip.ui.progressbar import BaseProgressBar, TextProgressBar
from qutip.cy.heom import cy_pad_csr
from qutip.cy.spmath import zcsr_kron
from qutip.fastsparse import fast_csr_matrix, fast_identity

ODE_FUNC_TYPES = (types.FunctionType, types.BuiltinFunctionType,
                  types.MethodType, types.BuiltinMethodType,  partial)


class HEOMSolver(object):
    """
    This is superclass for all solvers that use the HEOM method for
    calculating the dynamics evolution. There are many references for this.
    A good introduction, and perhaps closest to the notation used here is:
    DOI:10.1103/PhysRevLett.104.250401
    A more canonical reference, with full derivation is:
    DOI: 10.1103/PhysRevA.41.6676
    The method can compute open system dynamics without using any Markovian
    or rotating wave approximation (RWA) for systems where the bath
    correlations can be approximated to a sum of complex eponentials.
    The method builds a matrix of linked differential equations, which are
    then solved used the same ODE solvers as other qutip solvers (e.g. mesolve)

    This class should be treated as abstract. Currently the only subclass
    implemented is that for the Drude-Lorentz spectral density. This covers
    the majority of the work that has been done using this model, and there
    are some performance advantages to assuming this model where it is
    appropriate.

    There are opportunities to develop a more general spectral density code.

    Attributes
    ----------
    H_sys : :class:`qutip.Qobj`
        System Hamiltonian.
        Static Qobj or callback function returning Qobj
        NOTE: Assumes that H(t) is constant in each timeslice (of tlist)

    args : dict
        Dictionary of arguments passed to the time dependent Hamiltonian function

    coup_op : Qobj
        Operator describing the coupling between system and bath.

    coup_strength : float
        Coupling strength.

    temperature : float
        Bath temperature, in units corresponding to planck

    N_cut : int
        Cutoff parameter for the bath

    N_exp : int
        Number of exponential terms used to approximate the bath correlation
        functions

    planck : float
        reduced Planck constant

    boltzmann : float
        Boltzmann's constant

    options : :class:`qutip.solver.Options`
        Generic solver options.
        If set to None the default options will be used

    progress_bar: BaseProgressBar
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the simulation.

    stats : :class:`qutip.solver.Stats`
        optional container for holding performance statitics
        If None is set, then statistics are not collected
        There may be an overhead in collecting statistics

    exp_coeff : list of complex
        Coefficients for the exponential series terms

    exp_freq : list of complex
        Frequencies for the exponential series terms

    L_helems : :class:`fast_csr_matrix`
        The HEOM superoperator, without the system components
    """
    #TODO: Make ABC
    def __init__(self):
        raise NotImplementedError("This is a abstract class only. "
                "Use a subclass, for example HSolverDL")

    def reset(self):
        """
        Reset any attributes to default values
        """
        self.planck = 1.0
        self.boltzmann = 1.0
        self.H_sys = None
        self.args = None
        self.coup_op = None
        self.coup_strength = 0.0
        self.temperature = 1.0
        self.N_cut = 10
        self.N_exp = 2
        self.N_he = 0

        self.exp_coeff = None
        self.exp_freq = None

        self.td_type = None
        self.options = None
        self.progress_bar = None
        self.stats = None

        self.ode = None
        self.configured = False

        #TODO check is this used
        #self._td_H_sys = False

    def configure(self, H_sys, coup_op, coup_strength, temperature,
                     N_cut, N_exp, td_type=None, args=None,
                     planck=None, boltzmann=None,
                     renorm=None, bnd_cut_approx=None,
                     options=None, progress_bar=None, stats=None):
        """
        Configure the solver using the passed parameters
        The parameters are described in the class attributes, unless there
        is some specific behaviour

        Parameters
        ----------
        options : :class:`qutip.solver.Options`
            Generic solver options.
            If set to None the default options will be used

        progress_bar: BaseProgressBar
            Optional instance of BaseProgressBar, or a subclass thereof, for
            showing the progress of the simulation.
            If set to None, then the default progress bar will be used
            Set to False for no progress bar

        stats: :class:`qutip.solver.Stats`
            Optional instance of solver.Stats, or a subclass thereof, for
            storing performance statistics for the solver
            If set to True, then the default Stats for this class will be used
            Set to False for no stats
        """

        # check for time dependent components and set
        # dimensions set by system
        hsys_info = self._check_H_sys(H_sys)
        self.sys_dims = hsys_info['dims']
        if hsys_info['n_td'] == 0:
            if td_type is not None:
                warnings.warn("td_type specified as '{}', "
                              "but no td components".format(td_type))
                self.td_type = None
        else:
            #TODO add some compatibility checking here when string type
            # has been added
            # For now this just allows 'pwc' to be set
            #TODO need to document this td type stuff
            #note will have to use 2 properties, maybe
            # td_fmt_type='func'|'listfunc'|'str'
            # and td_func_type='pwc'|'continuous'
            if td_type is not None:
                self.td_type = td_type
            else:
                self.td_type = 'continuous'

#        self.H0 = None
#        if isinstance(H_sys, Qobj):
#            self.td_type = None
#            self.H0 = H_sys
#        elif isinstance(H_sys, (types.FunctionType,
#                            types.BuiltinFunctionType, partial)):
#            self.H_sys = H_sys
#            self.td_type = 'f'
#            self.H0 = self.H_sys(0.0, args)
#
#        if not isinstance(self.H0, Qobj):
#            raise TypeError("Invalid type {} for H_sys. Must be {} or callback "
#                            "function".format(type(H_sys), Qobj))
#        if not self.H0.isoper:
#            raise TypeError("H_sys must be (or return) a vaild Hamiltonian")

        self.args = args
        self.coup_op = coup_op
        self.coup_strength = coup_strength
        self.temperature = temperature
        self.N_cut = N_cut
        self.N_exp = N_exp
        if planck: self.planck = planck
        if boltzmann: self.boltzmann = boltzmann
        if isinstance(options, Options): self.options = options
        if isinstance(progress_bar, BaseProgressBar):
            self.progress_bar = progress_bar
        elif progress_bar == True:
            self.progress_bar = TextProgressBar()
        elif progress_bar == False:
            self.progress_bar = None
        if isinstance(stats, Stats):
            self.stats = stats
        elif stats == True:
            self.stats = self.create_new_stats()
        elif stats == False:
            self.stats = None



    def create_new_stats(self):
        """
        Creates a new stats object suitable for use with this solver
        Note: this solver expects the stats object to have sections
            config
            integrate
        """
        stats = Stats(['config', 'run'])
        stats.header = "Hierarchy Solver Stats"
        return stats

    def get_L_helems_info(self, L_helems=None):
        if L_helems is None:
            L = self.L_helems
        if L is None:
            return 'None'
        Ld = L.data
        Lda = np.abs(Ld)
        info = "shape: {}, nnz: {}".format(L.shape, L.nnz)

        if L.nnz > 0:
            info += ", min: {}, max: {}, avg(nz): {}".format(
                    np.min(Lda), np.max(Lda), np.mean(Lda))
        return info

    def _check_H_sys(self, H_sys):
        """
        Check that the format of H_sys is vaild. This is either constant
        or valid time-dependent format.

        Returns
        -------
        info : dict
            Dictionary with info about the system Hamiltonian.
            Including: 'dims' - system dimensions
            'n_comp' - number of components
            'n_td' - number of time-dependent components

        """
        str_Hsys_fmt_error = "H_sys must be a Qobj or supported td format"
        n_comp = 0
        n_td = 1
        dims = None
        def check_dims(chk_dims):
            nonlocal dims
            if dims is None:
                dims = chk_dims
            else:
                if dims != chk_dims:
                    raise ValueError("Incompatible dims for H_sys component"
                                     " {}".format(n_comp))
        if isinstance(H_sys, list):
            for h_comp in H_sys:
                if isinstance(h_comp, Qobj):
                    # Constant H component
                    H = h_comp
                elif isinstance(h_comp, list) and len(h_comp) == 2:
                    H = h_comp[0]
                    h_func = h_comp[1]
                    if not isinstance(h_comp[0], Qobj):
                        raise ValueError(str_Hsys_fmt_error)
                    if not isinstance(h_func, ODE_FUNC_TYPES):
                        raise TypeError("Invalid type {} 'h_func'. "
                                "Only function type time-dependence "
                                "currently supported".format(type(h_func)))
                    n_td += 1
                else:
                    raise ValueError(str_Hsys_fmt_error)
                check_dims(H.dims)
                n_comp += 1

        elif isinstance(H_sys, Qobj):
            dims = H_sys.dims
            n_comp = 1

        else:
            raise ValueError(str_Hsys_fmt_error)

        return {'dims': dims, 'n_comp': n_comp, 'n_td': n_td}

    def _configure_integ(self, unit_helems, L_helems, H_sys, options, stats):

        # format has already been checked in _check_H_sys, so just
        # make the L_list
        str_Hsys_fmt_error = ("H_sys must be a Qobj or supported td format."
                              "H_sys format should have been checked "
                              "in _check_H_sys!")
        if stats:
            start_integ_conf = timeit.default_timer()
            ss_conf = stats.sections.get('config')

        #TODO only copy L_helems if n_td > 0
        # for now just do it anyway, but its memory greedy
        n_td = 0
        L_const = L_helems.copy()
        L_list = [L_const]
        if isinstance(H_sys, list):
            # check for td elements
            for h_comp in H_sys:
                if isinstance(h_comp, Qobj):
                    # Constant H component, just add it
                    L_H = zcsr_kron(unit_helems, liouvillian(h_comp).data)
                    L_const += L_H
                elif isinstance(h_comp, list) and len(h_comp) == 2:
                    # time dependent H component
                    H = h_comp[0]
                    h_func = h_comp[1]
                    L_td = zcsr_kron(unit_helems, liouvillian(H).data)
                    L_list.append([L_td, h_func])
                    n_td += 1
                else:
                    raise ValueError(str_Hsys_fmt_error)

        elif isinstance(H_sys, Qobj):
            L_H = zcsr_kron(unit_helems, liouvillian(H_sys).data)
            L_helems += L_H

        else:
            raise ValueError(str_Hsys_fmt_error)


        if len(L_list) == 1:
            r = scipy.integrate.ode(cy_ode_rhs)
            r.set_f_params(L_const.data, L_const.indices, L_const.indptr)
        else:
            if self.td_type == 'pwc':
                L = _combineL(0, L_list)
                r = scipy.integrate.ode(cy_ode_rhs)
                r.set_f_params(L.data, L.indices, L.indptr)
                self._L_list = L_list
            else:
                r = scipy.integrate.ode(_dsuper_list_td)
                #TODO need to add args here
                r.set_f_params(L_list)

        r.set_integrator('zvode', method=options.method, order=options.order,
                         atol=options.atol, rtol=options.rtol,
                         nsteps=options.nsteps, first_step=options.first_step,
                         min_step=options.min_step, max_step=options.max_step)
        self._ode = r
        if stats:
            if n_td == 0:
                stats.add_message('integ td', 'constant', ss_conf)
            else:
                stats.add_message('integ td',
                                  '{} td components'.format(n_td), ss_conf)
            time_now = timeit.default_timer()
            stats.add_timing('Integrator config',
                             time_now - start_integ_conf,
                            ss_conf)

class HSolverDL(HEOMSolver):
    """
    HEOM solver based on the Drude-Lorentz model for spectral density.
    Drude-Lorentz bath the correlation functions can be exactly analytically
    expressed as an infinite sum of exponentials which depend on the
    temperature, these are called the Matsubara terms or Matsubara frequencies

    For practical computation purposes an approximation must be used based
    on a small number of Matsubara terms (typically < 4).

    Attributes
    ----------
    cut_freq : float
        Bath spectral density cutoff frequency.

    renorm : bool
        Apply renormalisation to coupling terms
        Can be useful if using SI units for planck and boltzmann

    bnd_cut_approx : bool
        Use boundary cut off approximation
        Can be
    """

    def __init__(self, H_sys, coup_op, coup_strength, temperature,
                     N_cut, N_exp, cut_freq, planck=1.0, boltzmann=1.0,
                     td_type=None, args=None,
                     renorm=True, bnd_cut_approx=True,
                     options=None, progress_bar=None, stats=None):

        self.reset()

        if options is None:
            self.options = Options()
        else:
            self.options = options

        self.progress_bar = False
        if progress_bar is None:
            self.progress_bar = BaseProgressBar()
        elif progress_bar == True:
            self.progress_bar = TextProgressBar()

        # the other attributes will be set in the configure method
        self.configure(H_sys, coup_op, coup_strength, temperature,
                     N_cut, N_exp, cut_freq, td_type=td_type, args=args,
                     planck=planck, boltzmann=boltzmann,
                     renorm=renorm, bnd_cut_approx=bnd_cut_approx, stats=stats)

    def reset(self):
        """
        Reset any attributes to default values
        """
        HEOMSolver.reset(self)
        self.cut_freq = 1.0
        self.renorm = False
        self.bnd_cut_approx = False

    def configure(self, H_sys, coup_op, coup_strength, temperature,
                     N_cut, N_exp, cut_freq, td_type=None, args=None,
                     planck=None, boltzmann=None,
                     renorm=None, bnd_cut_approx=None,
                     options=None, progress_bar=None, stats=None):
        """
        Calls configure from :class:`HEOMSolver` and sets any attributes
        that are specific to this subclass
        """
        start_config = timeit.default_timer()

        HEOMSolver.configure(self, H_sys, coup_op, coup_strength,
                    temperature, N_cut, N_exp, args=args,
                    planck=planck, boltzmann=boltzmann,
                    options=options, progress_bar=progress_bar, stats=stats)
        self.cut_freq = cut_freq
        if renorm is not None: self.renorm = renorm
        if bnd_cut_approx is not None: self.bnd_cut_approx = bnd_cut_approx

        # Load local values for optional parameters
        # Constants and Hamiltonian.
        hbar = self.planck
        options = self.options
        progress_bar = self.progress_bar
        stats = self.stats


        if stats:
            ss_conf = stats.sections.get('config')
            if ss_conf is None:
                ss_conf = stats.add_section('config')

        c, nu = self._calc_matsubara_params()

        if renorm:
            norm_plus, norm_minus = self._calc_renorm_factors()
            if stats:
                stats.add_message('options', 'renormalisation', ss_conf)

        N_temp = 1
        for i in self.sys_dims[0]:
            N_temp *= i
        sup_dim = N_temp**2
        unit_sys = qeye(N_temp)

        # Use shorthands (mainly as in referenced PRL)
        lam0 = self.coup_strength
        gam = self.cut_freq
        N_c = self.N_cut
        N_m = self.N_exp
        Q = coup_op # Q as shorthand for coupling operator
        beta = 1.0/(self.boltzmann*self.temperature)

        # Ntot is the total number of ancillary elements in the hierarchy
        # Ntot = factorial(N_c + N_m) / (factorial(N_c)*factorial(N_m))
        # Turns out to be the same as nstates from state_number_enumerate
        N_he, he2idx, idx2he = enr_state_dictionaries([N_c + 1]*N_m , N_c)

        unit_helems = fast_identity(N_he)

        if self.bnd_cut_approx:
            # the Tanimura boundary cut off operator
            if stats:
                stats.add_message('options', 'boundary cutoff approx', ss_conf)
            op = -2*spre(Q)*spost(Q.dag()) + spre(Q.dag()*Q) + spost(Q.dag()*Q)

            approx_factr = ((2*lam0 / (beta*gam*hbar)) - 1j*lam0) / hbar
            for k in range(N_m):
                approx_factr -= (c[k] / nu[k])
            L_bnd = -approx_factr*op.data
            L_helems = zcsr_kron(unit_helems, L_bnd)
        else:
            L_helems = fast_csr_matrix(shape=(N_he*sup_dim, N_he*sup_dim))

        # Build the hierarchy element interaction matrix
        if stats: start_helem_constr = timeit.default_timer()

        unit_sup = spre(unit_sys).data
        spreQ = spre(Q).data
        spostQ = spost(Q).data
        commQ = (spre(Q) - spost(Q)).data
        N_he_interact = 0

        for he_idx in range(N_he):
            print("L_helems nnz: ", L_helems.nnz)
            he_state = list(idx2he[he_idx])
            n_excite = sum(he_state)

            # The diagonal elements for the hierarchy operator
            # coeff for diagonal elements
            sum_n_m_freq = 0.0
            for k in range(N_m):
                sum_n_m_freq += he_state[k]*nu[k]

            op = -sum_n_m_freq*unit_sup
            L_he = cy_pad_csr(op, N_he, N_he, he_idx, he_idx)
            print("L_he nnz: ", L_he.nnz)
            L_helems += L_he

            # Add the neighour interations
            he_state_neigh = copy(he_state)
            for k in range(N_m):

                n_k = he_state[k]
                if n_k >= 1:
                    # find the hierarchy element index of the neighbour before
                    # this element, for this Matsubara term
                    he_state_neigh[k] = n_k - 1
                    he_idx_neigh = he2idx[tuple(he_state_neigh)]

                    op = c[k]*spreQ - np.conj(c[k])*spostQ
                    if renorm:
                        op = -1j*norm_minus[n_k, k]*op
                    else:
                        op = -1j*n_k*op

                    L_he = cy_pad_csr(op, N_he, N_he, he_idx, he_idx_neigh)
                    print("L_he- nnz: ", L_he.nnz)
                    L_helems += L_he
                    N_he_interact += 1

                    he_state_neigh[k] = n_k

                if n_excite <= N_c - 1:
                    # find the hierarchy element index of the neighbour after
                    # this element, for this Matsubara term
                    he_state_neigh[k] = n_k + 1
                    he_idx_neigh = he2idx[tuple(he_state_neigh)]

                    op = commQ
                    if renorm:
                        op = -1j*norm_plus[n_k, k]*op
                    else:
                        op = -1j*op

                    L_he = cy_pad_csr(op, N_he, N_he, he_idx, he_idx_neigh)
                    print("L_he+ nnz: ", L_he.nnz)
                    L_helems += L_he
                    N_he_interact += 1

                    he_state_neigh[k] = n_k

        # These are used when td_type = 'pwc'
        # and they may be useful to someone
        self.L_helems = L_helems

        if stats:
            stats.add_timing('hierarchy contruct',
                             timeit.default_timer() - start_helem_constr,
                            ss_conf)
            stats.add_count('Num hierarchy elements', N_he, ss_conf)
            stats.add_count('Num he interactions', N_he_interact, ss_conf)
            stats.add_message('L_helems info', self.get_L_helems_info(),
                              ss_conf)

        self._configure_integ(unit_helems, L_helems, H_sys, options, stats)

        if stats:
            time_now = timeit.default_timer()

            if ss_conf.total_time is None:
                ss_conf.total_time = time_now - start_config
            else:
                ss_conf.total_time += time_now - start_config

        self._N_he = N_he
        self._sup_dim = sup_dim

        self._configured = True
#
#    def _add_sys_liouvillian_helems(self, H, L_helems=None):
#        if L_helems is None:
#            L_helems = self._sysless_helems
#
#        H_he = zcsr_kron(self._unit_helems, liouvillian(H).data)
#        return L_helems + H_he


    def run(self, rho0, tlist):
        """
        Function to solve for an open quantum system using the
        HEOM model.

        Parameters
        ----------
        rho0 : Qobj
            Initial state (density matrix) of the system.

        tlist : list
            Time over which system evolves.

        Hlist : Qobj or list of Qobj
            Optional system Hamiltonian
            or list of system  Hamiltonians for each timeslot
            i.e. Hsys piecewise constant in the evolution

        Returns
        -------
        results : :class:`qutip.solver.Result`
            Object storing all results from the simulation.
        """

        start_run = timeit.default_timer()

        sup_dim = self._sup_dim
        stats = self.stats
        r = self._ode

        if not self._configured:
            raise RuntimeError("Solver must be configured before it is run")
        if stats:
            ss_conf = stats.sections.get('config')
            if ss_conf is None:
                raise RuntimeError("No config section for solver stats")
            ss_run = stats.sections.get('run')
            if ss_run is None:
                ss_run = stats.add_section('run')

        output = Result()
        output.solver = "hsolve"
        output.times = tlist
        output.states = []

        if stats: start_init = timeit.default_timer()
        output.states.append(Qobj(rho0))
        rho0_flat = rho0.full().ravel('F') # Using 'F' effectively transposes
        rho0_he = np.zeros([sup_dim*self._N_he], dtype=complex)
        rho0_he[:sup_dim] = rho0_flat
        r.set_initial_value(rho0_he, tlist[0])

        if stats:
            stats.add_timing('initialize',
                             timeit.default_timer() - start_init, ss_run)
            start_integ = timeit.default_timer()

        dt = np.diff(tlist)
        n_tsteps = len(tlist)
        for t_idx, t in enumerate(tlist):
            if t_idx < n_tsteps - 1:
                if self.td_type == 'pwc':
                    # Update the HEOM based on L(t)
                    # NOTE: assumes H constant in this timeslice
                    # and that the tlist coincicdes with timeslots
                    L = _combineL(t, self._L_list)
                    r.set_f_params(L.data, L.indices, L.indptr)
                # td_type == 'continuous' handled by the integrator

                r.integrate(r.t + dt[t_idx])
                rho = Qobj(r.y[:sup_dim].reshape(rho0.shape), dims=rho0.dims)
                output.states.append(rho)

        if stats:
            time_now = timeit.default_timer()
            stats.add_timing('integrate',
                             time_now - start_integ, ss_run)
            if ss_run.total_time is None:
                ss_run.total_time = time_now - start_run
            else:
                ss_run.total_time += time_now - start_run
            stats.total_time = ss_conf.total_time + ss_run.total_time

        return output

    def _calc_matsubara_params(self):
        """
        Calculate the Matsubara coefficents and frequencies

        Returns
        -------
        c, nu: both list(float)

        """
        c = []
        nu = []
        lam0 = self.coup_strength
        gam = self.cut_freq
        hbar = self.planck
        beta = 1.0/(self.boltzmann*self.temperature)
        N_m = self.N_exp

        g = 2*np.pi / (beta*hbar)
        for k in range(N_m):
            if k == 0:
                nu.append(gam)
                c.append(lam0*gam*
                    (1.0/np.tan(gam*hbar*beta/2.0) - 1j) / hbar)
            else:
                nu.append(k*g)
                c.append(4*lam0*gam*nu[k] /
                      ((nu[k]**2 - gam**2)*beta*hbar**2))

        self.exp_coeff = c
        self.exp_freq = nu
        return c, nu

    def _calc_renorm_factors(self):
        """
        Calculate the renormalisation factors

        Returns
        -------
        norm_plus, norm_minus : array[N_c, N_m] of float
        """
        c = self.exp_coeff
        N_m = self.N_exp
        N_c = self.N_cut

        norm_plus = np.empty((N_c+1, N_m))
        norm_minus = np.empty((N_c+1, N_m))
        for k in range(N_m):
            for n in range(N_c+1):
                norm_plus[n, k] = np.sqrt(abs(c[k])*(n + 1))
                norm_minus[n, k] = np.sqrt(float(n)/abs(c[k]))

        return norm_plus, norm_minus


def _pad_csr(A, row_scale, col_scale, insertrow=0, insertcol=0):
    """
    Expand the input csr_matrix to a greater space as given by the scale.
    Effectively inserting A into a larger matrix
         zeros([A.shape[0]*row_scale, A.shape[1]*col_scale]
    at the position [A.shape[0]*insertrow, A.shape[1]*insertcol]
    The same could be achieved through using a kron with a matrix with
    one element set to 1. However, this is more efficient
    """

    # ajgpitch 2016-03-08:
    # Clearly this is a very simple operation in dense matrices
    # It seems strange that there is nothing equivalent in sparse however,
    # after much searching most threads suggest directly addressing
    # the underlying arrays, as done here.
    # This certainly proved more efficient than other methods such as stacking
    #TODO: Perhaps cythonize and move to spmatfuncs

    if not isinstance(A, sp.csr_matrix):
        raise TypeError("First parameter must be a csr matrix")
    nrowin = A.shape[0]
    ncolin = A.shape[1]
    nrowout = nrowin*row_scale
    ncolout = ncolin*col_scale

    A._shape = (nrowout, ncolout)
    if insertcol == 0:
        pass
    elif insertcol > 0 and insertcol < col_scale:
        A.indices = A.indices + insertcol*ncolin
    else:
        raise ValueError("insertcol must be >= 0 and < col_scale")

    if insertrow == 0:
        A.indptr = np.concatenate((A.indptr,
                        np.array([A.indptr[-1]]*(row_scale-1)*nrowin)))
    elif insertrow == row_scale-1:
        A.indptr = np.concatenate((np.array([0]*(row_scale - 1)*nrowin),
                                   A.indptr))
    elif insertrow > 0 and insertrow < row_scale - 1:
         A.indptr = np.concatenate((np.array([0]*insertrow*nrowin), A.indptr,
                np.array([A.indptr[-1]]*(row_scale - insertrow - 1)*nrowin)))
    else:
        raise ValueError("insertrow must be >= 0 and < row_scale")

    return A

def _combineL(t, L_list):
    L_tot = None
    for l_comp in L_list:
        if isinstance(l_comp, list):
            L = l_comp[0] * l_comp[1](t)
        else:
            L = l_comp
        if L_tot is None:
            L_tot = L
        else:
            L_tot += L
    return L_tot

def _dsuper_list_td(t, y, L_list):#, args):

    return _ode_super_func(t, y, _combineL(t, L_list))

def _ode_super_func(t, y, data):
    #ym = vec2mat(y)
    #return (data*ym).ravel('F')
    #TODO use cy_ode_rhs?
    return (data*y)
