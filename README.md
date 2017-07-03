The New Crackpot Project
========================

Try to embed rules from discrete {0,1}^9 (9? or 8?) into continuous R^2, by analogy to RxnDfzn.
Use a network with some kind of embedding (pos nonlinear?) and a single "convolutional" filter,
with some feature engineering. Like first separate into a `sum` channel with

 1 1 1
 1 0 1
 1 1 1

and a `cell` channel with

 0 0 0
 0 1 0 
 0 0 0

both wrapped toroidally or zero padded, and then learn some kind of network on top of that which
will incorporate the rule embedding to try to approximate the next state.

Idea is to mimic word2vec, but here we have a completely deterministic (thus clearly Markovian)
system, simpler than a sentence in that way, but much higher dimensional. But the idea is to have
the rules be semantically embedded so that the dynamics can be smoothly connected (not interpolated,
but placed into neighborhoods).
