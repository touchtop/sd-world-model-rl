# Hopper Ability Entry Trace Report

本报告只读 retrospective train traces，用于追踪高质量能力首次进入 replay 分布的过程；输出不接入在线训练。

## First Crossings

- `seed1_R>1`: cycle=`4`, episode=`33`, best_reward=`6.549`, crossing_episodes=`1`
- `seed1_R>10`: cycle=`7`, episode=`63`, best_reward=`10.298`, crossing_episodes=`1`
- `seed1_R>50`: cycle=`12`, episode=`119`, best_reward=`130.184`, crossing_episodes=`6`
- `seed1_R>100`: cycle=`12`, episode=`119`, best_reward=`130.184`, crossing_episodes=`3`
- `seed2_R>1`: cycle=`17`, episode=`169`, best_reward=`5.213`, crossing_episodes=`1`
- `seed2_R>10`: cycle=`18`, episode=`179`, best_reward=`11.745`, crossing_episodes=`1`
- `seed2_R>50`: cycle=`25`, episode=`249`, best_reward=`88.417`, crossing_episodes=`3`
- `seed2_R>100`: cycle=`None`, episode=`None`, best_reward=`0.000`, crossing_episodes=`0`
- `seed3_R>1`: cycle=`8`, episode=`75`, best_reward=`2.768`, crossing_episodes=`2`
- `seed3_R>10`: cycle=`21`, episode=`209`, best_reward=`12.342`, crossing_episodes=`1`
- `seed3_R>50`: cycle=`23`, episode=`221`, best_reward=`56.352`, crossing_episodes=`1`
- `seed3_R>100`: cycle=`None`, episode=`None`, best_reward=`0.000`, crossing_episodes=`0`

## Cycle Summary

- seed `1` cycle `001`: mean=`0.017`, max=`0.174`, >1/10/50/100=`0/0/0/0`, rule_gate=`0.402`, transition_gate=`0.000`
- seed `1` cycle `002`: mean=`0.009`, max=`0.174`, >1/10/50/100=`0/0/0/0`, rule_gate=`0.670`, transition_gate=`0.000`
- seed `1` cycle `003`: mean=`0.015`, max=`0.276`, >1/10/50/100=`0/0/0/0`, rule_gate=`0.873`, transition_gate=`0.000`
- seed `1` cycle `004`: mean=`0.175`, max=`6.549`, >1/10/50/100=`1/0/0/0`, rule_gate=`1.015`, transition_gate=`0.000`
- seed `1` cycle `005`: mean=`0.157`, max=`6.549`, >1/10/50/100=`1/0/0/0`, rule_gate=`1.055`, transition_gate=`0.000`
- seed `1` cycle `006`: mean=`0.266`, max=`6.549`, >1/10/50/100=`3/0/0/0`, rule_gate=`1.104`, transition_gate=`0.000`
- seed `1` cycle `007`: mean=`0.375`, max=`10.298`, >1/10/50/100=`4/1/0/0`, rule_gate=`1.167`, transition_gate=`0.000`
- seed `1` cycle `008`: mean=`1.082`, max=`16.841`, >1/10/50/100=`10/4/0/0`, rule_gate=`1.309`, transition_gate=`0.000`
- seed `1` cycle `009`: mean=`1.874`, max=`20.970`, >1/10/50/100=`17/9/0/0`, rule_gate=`1.347`, transition_gate=`0.000`
- seed `1` cycle `010`: mean=`2.684`, max=`44.923`, >1/10/50/100=`21/13/0/0`, rule_gate=`1.516`, transition_gate=`0.021`
- seed `1` cycle `011`: mean=`3.951`, max=`46.486`, >1/10/50/100=`28/19/0/0`, rule_gate=`1.654`, transition_gate=`0.062`
- seed `1` cycle `012`: mean=`8.464`, max=`130.184`, >1/10/50/100=`36/26/6/3`, rule_gate=`1.649`, transition_gate=`0.072`
- seed `1` cycle `013`: mean=`15.503`, max=`145.867`, >1/10/50/100=`45/35/15/9`, rule_gate=`1.655`, transition_gate=`0.082`
- seed `1` cycle `014`: mean=`23.176`, max=`171.463`, >1/10/50/100=`53/43/23/17`, rule_gate=`1.743`, transition_gate=`0.124`
- seed `1` cycle `015`: mean=`32.586`, max=`199.547`, >1/10/50/100=`62/52/32/26`, rule_gate=`1.860`, transition_gate=`0.182`
- seed `1` cycle `016`: mean=`39.880`, max=`259.614`, >1/10/50/100=`70/59/39/33`, rule_gate=`1.870`, transition_gate=`0.195`
- seed `1` cycle `017`: mean=`48.144`, max=`316.565`, >1/10/50/100=`78/67/47/40`, rule_gate=`1.881`, transition_gate=`0.235`
- seed `1` cycle `018`: mean=`60.372`, max=`344.965`, >1/10/50/100=`87/76/56/49`, rule_gate=`1.806`, transition_gate=`0.260`
- seed `1` cycle `019`: mean=`72.291`, max=`372.833`, >1/10/50/100=`96/85/65/58`, rule_gate=`1.912`, transition_gate=`0.338`
- seed `1` cycle `020`: mean=`84.644`, max=`383.964`, >1/10/50/100=`105/94/74/67`, rule_gate=`2.111`, transition_gate=`0.425`
- seed `1` cycle `021`: mean=`97.467`, max=`468.795`, >1/10/50/100=`114/103/83/76`, rule_gate=`2.255`, transition_gate=`0.496`
- seed `1` cycle `022`: mean=`110.538`, max=`481.397`, >1/10/50/100=`123/112/92/85`, rule_gate=`2.272`, transition_gate=`0.520`
- seed `1` cycle `023`: mean=`124.064`, max=`511.689`, >1/10/50/100=`132/121/101/94`, rule_gate=`2.379`, transition_gate=`0.624`
- seed `1` cycle `024`: mean=`136.334`, max=`511.689`, >1/10/50/100=`141/130/110/103`, rule_gate=`2.395`, transition_gate=`0.658`
- seed `1` cycle `025`: mean=`148.512`, max=`530.227`, >1/10/50/100=`150/139/119/112`, rule_gate=`2.414`, transition_gate=`0.665`
- seed `2` cycle `001`: mean=`0.000`, max=`0.000`, >1/10/50/100=`0/0/0/0`, rule_gate=`0.552`, transition_gate=`0.000`
- seed `2` cycle `002`: mean=`0.000`, max=`0.008`, >1/10/50/100=`0/0/0/0`, rule_gate=`1.089`, transition_gate=`0.000`
- seed `2` cycle `003`: mean=`0.004`, max=`0.101`, >1/10/50/100=`0/0/0/0`, rule_gate=`1.311`, transition_gate=`0.000`
- seed `2` cycle `004`: mean=`0.003`, max=`0.101`, >1/10/50/100=`0/0/0/0`, rule_gate=`1.440`, transition_gate=`0.000`
- seed `2` cycle `005`: mean=`0.002`, max=`0.101`, >1/10/50/100=`0/0/0/0`, rule_gate=`1.452`, transition_gate=`0.000`
- seed `2` cycle `006`: mean=`0.002`, max=`0.101`, >1/10/50/100=`0/0/0/0`, rule_gate=`1.479`, transition_gate=`0.000`
- seed `2` cycle `007`: mean=`0.002`, max=`0.101`, >1/10/50/100=`0/0/0/0`, rule_gate=`1.617`, transition_gate=`0.000`
- seed `2` cycle `008`: mean=`0.002`, max=`0.101`, >1/10/50/100=`0/0/0/0`, rule_gate=`1.816`, transition_gate=`0.000`
- seed `2` cycle `009`: mean=`0.002`, max=`0.101`, >1/10/50/100=`0/0/0/0`, rule_gate=`1.811`, transition_gate=`0.000`
- seed `2` cycle `010`: mean=`0.002`, max=`0.101`, >1/10/50/100=`0/0/0/0`, rule_gate=`1.957`, transition_gate=`0.003`
- seed `2` cycle `011`: mean=`0.002`, max=`0.101`, >1/10/50/100=`0/0/0/0`, rule_gate=`2.111`, transition_gate=`0.044`
- seed `2` cycle `012`: mean=`0.002`, max=`0.101`, >1/10/50/100=`0/0/0/0`, rule_gate=`2.103`, transition_gate=`0.047`
- seed `2` cycle `013`: mean=`0.002`, max=`0.101`, >1/10/50/100=`0/0/0/0`, rule_gate=`2.060`, transition_gate=`0.048`
- seed `2` cycle `014`: mean=`0.003`, max=`0.180`, >1/10/50/100=`0/0/0/0`, rule_gate=`2.088`, transition_gate=`0.065`
- seed `2` cycle `015`: mean=`0.004`, max=`0.180`, >1/10/50/100=`0/0/0/0`, rule_gate=`2.171`, transition_gate=`0.083`
- seed `2` cycle `016`: mean=`0.004`, max=`0.180`, >1/10/50/100=`0/0/0/0`, rule_gate=`2.111`, transition_gate=`0.085`
- seed `2` cycle `017`: mean=`0.034`, max=`5.213`, >1/10/50/100=`1/0/0/0`, rule_gate=`2.050`, transition_gate=`0.086`
- seed `2` cycle `018`: mean=`0.098`, max=`11.745`, >1/10/50/100=`2/1/0/0`, rule_gate=`2.046`, transition_gate=`0.090`
- seed `2` cycle `019`: mean=`0.294`, max=`13.390`, >1/10/50/100=`6/3/0/0`, rule_gate=`2.189`, transition_gate=`0.106`
- seed `2` cycle `020`: mean=`0.799`, max=`17.926`, >1/10/50/100=`13/9/0/0`, rule_gate=`2.374`, transition_gate=`0.123`
- seed `2` cycle `021`: mean=`1.088`, max=`17.926`, >1/10/50/100=`18/14/0/0`, rule_gate=`2.445`, transition_gate=`0.142`
- seed `2` cycle `022`: mean=`1.428`, max=`17.926`, >1/10/50/100=`24/20/0/0`, rule_gate=`2.500`, transition_gate=`0.172`
- seed `2` cycle `023`: mean=`1.691`, max=`25.097`, >1/10/50/100=`28/24/0/0`, rule_gate=`2.563`, transition_gate=`0.188`
- seed `2` cycle `024`: mean=`1.830`, max=`25.097`, >1/10/50/100=`32/27/0/0`, rule_gate=`2.556`, transition_gate=`0.183`
- seed `2` cycle `025`: mean=`2.763`, max=`88.417`, >1/10/50/100=`37/32/3/0`, rule_gate=`2.587`, transition_gate=`0.187`
- seed `3` cycle `001`: mean=`0.000`, max=`0.004`, >1/10/50/100=`0/0/0/0`, rule_gate=`0.549`, transition_gate=`0.000`
- seed `3` cycle `002`: mean=`0.001`, max=`0.012`, >1/10/50/100=`0/0/0/0`, rule_gate=`0.844`, transition_gate=`0.000`
- seed `3` cycle `003`: mean=`0.001`, max=`0.012`, >1/10/50/100=`0/0/0/0`, rule_gate=`1.001`, transition_gate=`0.000`
- seed `3` cycle `004`: mean=`0.002`, max=`0.024`, >1/10/50/100=`0/0/0/0`, rule_gate=`1.192`, transition_gate=`0.000`
- seed `3` cycle `005`: mean=`0.002`, max=`0.024`, >1/10/50/100=`0/0/0/0`, rule_gate=`1.223`, transition_gate=`0.000`
- seed `3` cycle `006`: mean=`0.003`, max=`0.032`, >1/10/50/100=`0/0/0/0`, rule_gate=`1.194`, transition_gate=`0.000`
- seed `3` cycle `007`: mean=`0.003`, max=`0.032`, >1/10/50/100=`0/0/0/0`, rule_gate=`1.230`, transition_gate=`0.000`
- seed `3` cycle `008`: mean=`0.051`, max=`2.768`, >1/10/50/100=`2/0/0/0`, rule_gate=`1.382`, transition_gate=`0.000`
- seed `3` cycle `009`: mean=`0.115`, max=`4.292`, >1/10/50/100=`4/0/0/0`, rule_gate=`1.403`, transition_gate=`0.000`
- seed `3` cycle `010`: mean=`0.180`, max=`5.053`, >1/10/50/100=`6/0/0/0`, rule_gate=`1.548`, transition_gate=`0.008`
- seed `3` cycle `011`: mean=`0.221`, max=`5.053`, >1/10/50/100=`8/0/0/0`, rule_gate=`1.658`, transition_gate=`0.038`
- seed `3` cycle `012`: mean=`0.253`, max=`5.053`, >1/10/50/100=`10/0/0/0`, rule_gate=`1.655`, transition_gate=`0.043`
- seed `3` cycle `013`: mean=`0.292`, max=`5.053`, >1/10/50/100=`12/0/0/0`, rule_gate=`1.631`, transition_gate=`0.046`
- seed `3` cycle `014`: mean=`0.323`, max=`5.053`, >1/10/50/100=`14/0/0/0`, rule_gate=`1.661`, transition_gate=`0.057`
- seed `3` cycle `015`: mean=`0.355`, max=`5.053`, >1/10/50/100=`16/0/0/0`, rule_gate=`1.729`, transition_gate=`0.069`
- seed `3` cycle `016`: mean=`0.356`, max=`5.053`, >1/10/50/100=`18/0/0/0`, rule_gate=`1.716`, transition_gate=`0.068`
- seed `3` cycle `017`: mean=`0.391`, max=`7.588`, >1/10/50/100=`20/0/0/0`, rule_gate=`1.727`, transition_gate=`0.071`
- seed `3` cycle `018`: mean=`0.397`, max=`7.588`, >1/10/50/100=`21/0/0/0`, rule_gate=`1.685`, transition_gate=`0.070`
- seed `3` cycle `019`: mean=`0.423`, max=`7.588`, >1/10/50/100=`23/0/0/0`, rule_gate=`1.777`, transition_gate=`0.081`
- seed `3` cycle `020`: mean=`0.441`, max=`7.588`, >1/10/50/100=`25/0/0/0`, rule_gate=`1.938`, transition_gate=`0.089`
- seed `3` cycle `021`: mean=`0.561`, max=`12.342`, >1/10/50/100=`28/1/0/0`, rule_gate=`1.994`, transition_gate=`0.102`
- seed `3` cycle `022`: mean=`1.207`, max=`38.903`, >1/10/50/100=`33/6/0/0`, rule_gate=`2.076`, transition_gate=`0.130`
- seed `3` cycle `023`: mean=`1.797`, max=`56.352`, >1/10/50/100=`38/9/1/0`, rule_gate=`2.115`, transition_gate=`0.145`
- seed `3` cycle `024`: mean=`2.071`, max=`62.079`, >1/10/50/100=`40/11/2/0`, rule_gate=`2.097`, transition_gate=`0.144`
- seed `3` cycle `025`: mean=`2.852`, max=`62.079`, >1/10/50/100=`46/16/4/0`, rule_gate=`2.199`, transition_gate=`0.162`

## Entry Window Contrasts

- `R>1_cycle004_entry_success`: windows=`20`, reward/future_reward=`1.540/1.153`, rootx/future_rootx=`0.144/0.208`, vx/future_vx=`0.490/0.329`, dominant_actions=`{'ankle_neg': 8, 'hip_neg': 5, 'knee_pos': 3, 'knee_neg': 2, 'hip_pos': 1, 'waist_neg': 1}`
- `R>1_cycle004_matched_failure`: windows=`47`, reward/future_reward=`0.020/0.000`, rootx/future_rootx=`0.031/-0.128`, vx/future_vx=`0.066/-0.205`, dominant_actions=`{'hip_neg': 25, 'waist_pos': 8, 'ankle_pos': 5, 'waist_neg': 5, 'hip_pos': 2, 'knee_pos': 1, 'ankle_neg': 1}`
- `R>1_cycle017_entry_success`: windows=`19`, reward/future_reward=`1.467/0.973`, rootx/future_rootx=`0.061/0.123`, vx/future_vx=`0.174/0.224`, dominant_actions=`{'hip_pos': 5, 'waist_neg': 5, 'ankle_pos': 4, 'ankle_neg': 3, 'knee_neg': 1, 'waist_pos': 1}`
- `R>1_cycle008_entry_success`: windows=`29`, reward/future_reward=`1.577/0.179`, rootx/future_rootx=`0.124/0.032`, vx/future_vx=`0.447/0.035`, dominant_actions=`{'ankle_neg': 9, 'hip_neg': 7, 'hip_pos': 7, 'waist_pos': 4, 'knee_neg': 2}`
- `R>1_cycle008_matched_failure`: windows=`23`, reward/future_reward=`0.035/0.000`, rootx/future_rootx=`0.027/-0.100`, vx/future_vx=`0.046/-0.155`, dominant_actions=`{'waist_pos': 9, 'waist_neg': 4, 'hip_neg': 4, 'ankle_pos': 3, 'hip_pos': 3}`
- `R>10_cycle007_entry_success`: windows=`26`, reward/future_reward=`5.315/2.769`, rootx/future_rootx=`0.227/0.115`, vx/future_vx=`0.760/0.162`, dominant_actions=`{'knee_neg': 9, 'ankle_neg': 6, 'waist_neg': 6, 'hip_pos': 4, 'waist_pos': 1}`
- `R>10_cycle007_matched_failure`: windows=`51`, reward/future_reward=`0.297/0.028`, rootx/future_rootx=`0.056/-0.097`, vx/future_vx=`0.153/-0.150`, dominant_actions=`{'hip_neg': 23, 'waist_pos': 10, 'hip_pos': 7, 'ankle_pos': 4, 'waist_neg': 4, 'knee_neg': 2, 'ankle_neg': 1}`
- `R>10_cycle018_entry_success`: windows=`25`, reward/future_reward=`2.868/2.555`, rootx/future_rootx=`0.130/0.175`, vx/future_vx=`0.412/0.299`, dominant_actions=`{'hip_pos': 6, 'waist_neg': 5, 'ankle_pos': 5, 'ankle_neg': 3, 'knee_neg': 3, 'waist_pos': 1, 'knee_pos': 1, 'hip_neg': 1}`
- `R>10_cycle018_matched_failure`: windows=`33`, reward/future_reward=`4.730/1.039`, rootx/future_rootx=`0.199/0.111`, vx/future_vx=`0.727/0.144`, dominant_actions=`{'ankle_neg': 23, 'hip_pos': 6, 'ankle_pos': 3, 'knee_pos': 1}`
- `R>10_cycle021_entry_success`: windows=`26`, reward/future_reward=`5.607/1.370`, rootx/future_rootx=`0.218/0.125`, vx/future_vx=`0.760/0.183`, dominant_actions=`{'hip_pos': 9, 'ankle_neg': 9, 'knee_pos': 3, 'waist_neg': 3, 'ankle_pos': 1, 'waist_pos': 1}`
- `R>50_cycle012_entry_success`: windows=`36`, reward/future_reward=`7.724/4.850`, rootx/future_rootx=`0.262/0.252`, vx/future_vx=`0.882/0.401`, dominant_actions=`{'knee_neg': 12, 'ankle_neg': 11, 'waist_neg': 6, 'hip_pos': 5, 'ankle_pos': 2}`
- `R>50_cycle012_matched_failure`: windows=`58`, reward/future_reward=`2.328/0.312`, rootx/future_rootx=`0.121/0.039`, vx/future_vx=`0.431/0.042`, dominant_actions=`{'ankle_neg': 27, 'waist_pos': 15, 'hip_pos': 5, 'waist_neg': 4, 'ankle_pos': 3, 'hip_neg': 3, 'knee_neg': 1}`
- `R>50_cycle025_entry_success`: windows=`24`, reward/future_reward=`6.530/3.997`, rootx/future_rootx=`0.249/0.180`, vx/future_vx=`0.829/0.270`, dominant_actions=`{'hip_pos': 8, 'waist_neg': 5, 'ankle_pos': 4, 'ankle_neg': 4, 'waist_pos': 2, 'knee_neg': 1}`
- `R>50_cycle023_entry_success`: windows=`21`, reward/future_reward=`6.805/1.668`, rootx/future_rootx=`0.244/0.152`, vx/future_vx=`0.862/0.227`, dominant_actions=`{'hip_pos': 12, 'ankle_neg': 4, 'waist_neg': 3, 'waist_pos': 2}`
- `R>50_cycle023_matched_failure`: windows=`36`, reward/future_reward=`5.908/2.307`, rootx/future_rootx=`0.198/0.138`, vx/future_vx=`0.665/0.212`, dominant_actions=`{'hip_pos': 15, 'waist_pos': 10, 'ankle_neg': 5, 'waist_neg': 4, 'ankle_pos': 2}`
- `R>100_cycle012_entry_success`: windows=`36`, reward/future_reward=`7.724/4.850`, rootx/future_rootx=`0.262/0.252`, vx/future_vx=`0.882/0.401`, dominant_actions=`{'knee_neg': 12, 'ankle_neg': 11, 'waist_neg': 6, 'hip_pos': 5, 'ankle_pos': 2}`
- `R>100_cycle012_matched_failure`: windows=`58`, reward/future_reward=`2.328/0.312`, rootx/future_rootx=`0.121/0.039`, vx/future_vx=`0.431/0.042`, dominant_actions=`{'ankle_neg': 27, 'waist_pos': 15, 'hip_pos': 5, 'waist_neg': 4, 'ankle_pos': 3, 'hip_neg': 3, 'knee_neg': 1}`

## Candidate Action Contrasts

- R>`1` cycle `004` `ankle_neg`: success `8` (0.400) vs failure `1` (0.021), rate_margin=`0.379`, future_reward_margin=`1.634`
- R>`10` cycle `021` `hip_pos`: success `9` (0.346) vs failure `0` (0.000), rate_margin=`0.346`, future_reward_margin=`1.227`
- R>`10` cycle `021` `ankle_neg`: success `9` (0.346) vs failure `0` (0.000), rate_margin=`0.346`, future_reward_margin=`0.978`
- R>`50` cycle `025` `hip_pos`: success `8` (0.333) vs failure `0` (0.000), rate_margin=`0.333`, future_reward_margin=`2.647`
- R>`50` cycle `012` `knee_neg`: success `12` (0.333) vs failure `1` (0.017), rate_margin=`0.316`, future_reward_margin=`6.699`
- R>`100` cycle `012` `knee_neg`: success `12` (0.333) vs failure `1` (0.017), rate_margin=`0.316`, future_reward_margin=`6.699`
- R>`1` cycle `008` `ankle_neg`: success `9` (0.310) vs failure `0` (0.000), rate_margin=`0.310`, future_reward_margin=`0.255`
- R>`10` cycle `007` `knee_neg`: success `9` (0.346) vs failure `2` (0.039), rate_margin=`0.307`, future_reward_margin=`2.432`
- R>`1` cycle `017` `waist_neg`: success `5` (0.263) vs failure `0` (0.000), rate_margin=`0.263`, future_reward_margin=`2.817`
- R>`1` cycle `017` `hip_pos`: success `5` (0.263) vs failure `0` (0.000), rate_margin=`0.263`, future_reward_margin=`0.879`
- R>`10` cycle `007` `ankle_neg`: success `6` (0.231) vs failure `1` (0.020), rate_margin=`0.211`, future_reward_margin=`1.089`
- R>`1` cycle `017` `ankle_pos`: success `4` (0.211) vs failure `0` (0.000), rate_margin=`0.211`, future_reward_margin=`0.000`
- R>`50` cycle `025` `waist_neg`: success `5` (0.208) vs failure `0` (0.000), rate_margin=`0.208`, future_reward_margin=`8.741`
- R>`10` cycle `018` `waist_neg`: success `5` (0.200) vs failure `0` (0.000), rate_margin=`0.200`, future_reward_margin=`2.506`
- R>`50` cycle `025` `ankle_neg`: success `4` (0.167) vs failure `0` (0.000), rate_margin=`0.167`, future_reward_margin=`6.441`
- R>`50` cycle `025` `ankle_pos`: success `4` (0.167) vs failure `0` (0.000), rate_margin=`0.167`, future_reward_margin=`0.150`
- R>`1` cycle `017` `ankle_neg`: success `3` (0.158) vs failure `0` (0.000), rate_margin=`0.158`, future_reward_margin=`0.000`
- R>`50` cycle `023` `hip_pos`: success `12` (0.571) vs failure `15` (0.417), rate_margin=`0.155`, future_reward_margin=`-1.090`
- R>`10` cycle `007` `waist_neg`: success `6` (0.231) vs failure `4` (0.078), rate_margin=`0.152`, future_reward_margin=`5.995`
- R>`1` cycle `004` `knee_pos`: success `3` (0.150) vs failure `1` (0.021), rate_margin=`0.129`, future_reward_margin=`0.000`
- R>`10` cycle `018` `knee_neg`: success `3` (0.120) vs failure `0` (0.000), rate_margin=`0.120`, future_reward_margin=`7.843`
- R>`10` cycle `021` `waist_neg`: success `3` (0.115) vs failure `0` (0.000), rate_margin=`0.115`, future_reward_margin=`5.264`
- R>`10` cycle `021` `knee_pos`: success `3` (0.115) vs failure `0` (0.000), rate_margin=`0.115`, future_reward_margin=`0.000`
- R>`1` cycle `008` `hip_pos`: success `7` (0.241) vs failure `3` (0.130), rate_margin=`0.111`, future_reward_margin=`0.415`
- R>`10` cycle `018` `ankle_pos`: success `5` (0.200) vs failure `3` (0.091), rate_margin=`0.109`, future_reward_margin=`0.000`
- R>`1` cycle `004` `knee_neg`: success `2` (0.100) vs failure `0` (0.000), rate_margin=`0.100`, future_reward_margin=`4.997`
- R>`50` cycle `012` `waist_neg`: success `6` (0.167) vs failure `4` (0.069), rate_margin=`0.098`, future_reward_margin=`3.245`
- R>`100` cycle `012` `waist_neg`: success `6` (0.167) vs failure `4` (0.069), rate_margin=`0.098`, future_reward_margin=`3.245`
- R>`50` cycle `025` `waist_pos`: success `2` (0.083) vs failure `0` (0.000), rate_margin=`0.083`, future_reward_margin=`2.340`
- R>`1` cycle `008` `knee_neg`: success `2` (0.069) vs failure `0` (0.000), rate_margin=`0.069`, future_reward_margin=`0.000`

## Interpretation

这些候选只表示入口窗口中的观察性差异。后续若转为规则，需要再经过 SD/TVA 反事实推演、跨周期后果验证和在线 shadow 审计。