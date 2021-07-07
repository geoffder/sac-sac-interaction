from neuron import h


class BasicSyn:
    def __init__(self, sec, pos, tau1, tau2, reversal, delay, weight):
        sec.push()
        self.syn = h.Exp2Syn(pos)
        self.syn.tau1 = tau1
        self.syn.tau2 = tau2
        self.syn.e = reversal
        self.con = h.NetCon(None, self.syn, 0., delay, weight)
        h.pop_section()

    @property
    def tau1(self):
        return self.syn.tau1

    @property
    def tau2(self):
        return self.syn.tau2

    @property
    def reversal(self):
        return self.syn.e

    @property
    def delay(self):
        return self.con.delay

    @property
    def weight(self):
        return self.con.weight[0]

    @tau1.setter
    def tau1(self, v):
        self.syn.tau1 = v

    @tau2.setter
    def tau2(self, v):
        self.syn.tau2 = v

    @reversal.setter
    def reversal(self, v):
        self.syn.e = v

    @delay.setter
    def delay(self, v):
        self.con.delay = v

    @weight.setter
    def weight(self, v):
        self.con.weight[0] = v
