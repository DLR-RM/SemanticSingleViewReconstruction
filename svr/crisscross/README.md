# Crisscross

Crisscross is the connection library used as a backend connecting the different networks.
This enables the usage of our models on different machines, which are in the same network.

### Core

It contains the `crisscross.Server`, `crisscross.Client` and the base class `crisscross.ConnectionInterface`. 
These can be used to set up your own easy to use server client architecture. 
These are designed to even work if one of them dies unplanned, and then just continue to wait until the `crisscross.Server` reboots.

The server is able to receive a `crisscross.Message` from the client, a derived version should also receive derived messages.

