import numpy as np
from src.env.beer_game_env import BeerGameEnv, BeerGameEnvSequential

def test_sequential_matches_joint_cycle():
    actions=[0,3,8,2,5,1,7,4]
    joint=BeerGameEnv(seed=19); seq=BeerGameEnvSequential(seed=19)
    joint.reset(seed=19); seq.reset(seed=19)
    for i in range(0,len(actions),4):
        a=np.asarray(actions[i:i+4],dtype=int)
        jo,jr,jt,jtr,ji=joint.step(a)
        for j in a[:-1]:
            so,sr,st,ss,si=seq.step(int(j)); assert sr==0 and si['substep']
        so,sr,st,s_trunc,str_=seq.step(int(a[-1]))
        assert np.allclose(so, np.r_[jo,0.0])
        assert np.isclose(sr,jr); assert st==jt; assert str_['substep']==False
