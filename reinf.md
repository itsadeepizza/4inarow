

# 1. (Nick) Obiettivi: imparare il reinforcement learning con il forza 4 + spiegazione del gioco

Spiegare molto grosso modo qual é l'obiettivo del RL (ma non come funziona)

Dare degli esmpi di applicazione del reinforcement learning (alphago, robot, jobshop problem)

Spiegare le regole del forza 4.
      


# 2. (Paolo con esempio gelato) Spiegazione veloce del deep Q learning

Come funziona il Q-learning:
- approccio naif e perché non funziona
- esempio gelato parte 1
- introduzione dei vari concetti della formula di bellman usando il gelato
- esempio del gelato parte 2
- Spiegazione della formula di bellman
- Come utilizzare la formula di bellman per allenare un NN


So now our goal is to train a model for making it play to connect4. But for training a model -a 
neural network or more generally every model in machine learning- we need, above all, a loss function.

The loss function always play a very important role, maybe the most important, as it defines
which function the model will approximate when it will be trained.
In most cases (for classification tasks for example) the loss function 
is simple and pretty intuitive (error in classification), but for reinforcement learning things are pretty trickier.

So the question is: which could be an adequate loss function for a model playing a game?


## The naif/classification approach

Let's start with a first (wrong) guess. An easy solution could be to proceed as for a classification task. So the model would take as input the state of the board, and as output the move to play. As there are 7 possible moves, the output should be something like this:

STATE (a tensor representing the board state) -> MODEL -> [0, 0, 0, 1, 0, 0, 0]

In this case the model would choose the central column (with a value equal 1). 

And what about the loss?

Ideally the loss should be high for wrong moves and low for good moves. But how to define what a "good move" is? 

More generally, how can we validate a strategy or any choice in our life? Unfortunately there is only one sure way: wait until the end and look to the results. 
This is as true in reinforcement learning as in every other contest, but it is quite impractical.
If we wait until the end of the game, we are forced to accumulate gradients during all the moves, and calculate a unique total loss.

How could a model be able to learn anything with only a loss for the whole game? How to find the contribution of a move to the beginning of the game to the actual state? This is just impossible (or maybe theoretically possible for a very very long training but practically unfeasible). It will be almost the same as decide all possible strategies at the beginning of the game, before the opponent make any move! 

So we need to calculate our loss in a different way. The main problem in the "naif" method is that we could not compress a game in one loss. So we need a different approach, where we decompose the game move after move, in multiple losses. Rather than shallow the whole game, we have to give multiple bites, and capitalise knowledge at each step. 

But how to achieve this result? We are lucky, there is already a solution ready for us: the **Q-learning**.

## Q-learning


The main idea is to assign, for each possible state of the board, and for each possible move, a score.

We call $Q$ this score, as for the "quality" of the move.

This score tell us how likely this move can provide us a good reward on the long term. So in order to define $Q$, we need firstly to define the reward $R$ and the discount factor $\gamma$

Let's consider the following example. Bill is sitting idle in his armchair, when suddenly an idea flashes into his head: what about an ice cream?

**Disegno bill seduto che pensa al gelato**

A decision has to be taken: what to do now? Does the reward associate to the ice-cream worth the uncomfort of leaving the  armchair and walk to the freezer?

Let's make the math.

First, for applying Q-learning we need to _discretize_ the process, i.e. defining a minimal time step. Let's say 10 seconds.

Now, we need to associate an immediate 






# 3.  - Tradeoff exploration - exploitation: vogliamo la miglior soluzione ma vogliamo esplorare diverse soluzioni
  
 
in the reinforcement learning field a key concept is represented by the tradeoff 
   between exploration and exploitation. Training a policy to be able to output always 
   the best action in a given state would need it to be always "greedy" for the action value, 
   but a greedy model would never explore and this means that a lot of potentially valuable actions 
   are just ignored. 
   
 ##  3.1 $\varepsilon$-greedy
   
In Reinforcement Learning, the agent or decision-maker, which interacts with the environment, is capable to learn what to do, this is, how to map situations to actions, in order to maximize a reward. 
The agent is not explicitly aware of which action to take, but instead must discover which action can lead the most reward through trial and error.
By going through the simulation it is possible to understand how the algorithm (agent) explores the action space and exploits the knowledge at any given time step. This tradeoff is fundamental to many RL algorithms. 
Exploration allows an agent to enhance its current knowledge about each action, hopefully leading to long-term benefit. Improving the accuracy of the estimated action-values, enables an agent to make more informed decisions in the future.
Exploitation, on the other hand, adopts the greedy action to get the most reward by exploiting the agent’s current action-value estimates. But by being greedy with respect to action-value estimates, may not actually get the most reward and lead to sub-optimal behaviour.
Epsilon-Greedy is a simple method to balance exploration and exploitation by choosing between exploration and exploitation randomly.
Exploitation generally means taking advantage of the currently available results, while exploration emphasizes the search for new solutions. Striking a delicate balance between the two can lead to better results.
   
   
   
  ### 3.2 separazione tra modello per il target e per la policy (spiegare pure cosa sono) *off-policy* et *on-policy*
  ### 3.2.1) Aggiornamento del target model: per aggiornare il target model si puo usare uno step fisso, oppure uno score (*mirror score*)
   
 A policy could be trained to be both explorative and eploitative at the same 
   time but by definition this means that some times this kind of policy will not choose the best 
   action in order to explore the others. A solution for this problem is provided by "off-policy" 
   algorithms, a class of algorithms (such as q-learning) that use different policies for different 
   scopes, in particular some policies are more explorative and some others are exploitative. 
   
After the training we want to have a policy that after some exploration has found a good strategy 
to exploit for maximizing the total reward, this is called the "target" policy. In this example 
the target policy is found evaluating and training epsilon-greedy policies which will explore less 
as time passes and hence they will become more greedy, during the training the policy's parameters 
will be copied to the target policy and this is how the target policy is computed. 
An additional step is added before the copy and it is the evaluation step, the policy's performance 
is evaluated using the "mirror score" and compared to the one of the current target policy, 
the copy will take place overwriting the old parameters only if the score is higher which means 
that the new target is better than the current one. This step ensure the new target to be strictly 
better after every replacement
   


# 4. (Bruno) Come il Q-learning si declina nel nostro caso


## 4.1 Spiegazione generale:(no dettagli)




## (Saverio) 4.2 Classe BatchBoard 

- Varie partite sono giocate in parallelo
- L'insieme degli stati per le varie partite è memorizzato su un tensore
- Ad ogni step una mossa viene selezionata per ogni board nel batch e giocata
- Per ogni giocata, la ricompense (e la loss) viene calcolata per ogni board nel batch
- Quando le partite finiscono per una board nel batch, questa si reinizializza senza modificare le altre
- Aggiungere estratti di codice




## 4.3 (Nick) Architettura conv contro linear layer

## 4.4 (Pietro) Il greedy player per allenare (citazione a progetto simile Fabio)

### 4.4.1 Perché un algoritmo greedy (che è diverso da $\varepsilon$-greedy!)?
Se il modello viene fatto giocare completamente a caso, non impare molt obene e si "disperde".
Bisogna quindi diminuire il lato esplorativo, inserendo nel training delle mosse che abbiano un minimo senso.
Quindi se nel $\varepsilon$-greedy abbiamo una percentuale di mosse giocate a caso, ora avremo una percentuale di mosse giocate con un algoritmo greedy.
- Una parte di mosse decise dal target - all'inizio del gioco fa schifo
- Una parte di mosse fatte a caso - fa sempre schifo
- Aggiunto alla fine -> Una parte di mosse greedy - all'inizio del gioco è molto forte
   
Le percentuali relative possono variare col tempo

### 4.4.2 Come funziona l'algoritmo greedy?
Che cos'è un algoritmo greedy? È un algoritmo che cerca sempre il vantaggio immediato e se ne frega del lungo termine (caso particolare di RL ma con gamma=0).

Che cos'è il vantaggio immediato a forza 4?
Lo calcoliamo prendendo quante "terzine aperte" (ossia 3 gettoni dello stesso colore seguiti da un buco, di fila e che potenzialmente possono essere completate) ha il giocatore e l'avversario.
L'algoritmo greedy gioca la mossa che massimizza il numero nelle proprie terzine (e ancora più delle quartine, ossia vince se può farlo) e diminuisce il numero delle terzine avversarie (blocca lavversari ose può farlo)

L'idea viene dall'Articolo Fabio ->   [https://towardsdatascience.com/playing-connect-4-with-deep-q-learning-76271ed663ca](https://)



## (Pietro) 4.5 Il mirror score

player è il giocatore del quale vuogliamo misurare la forza
n_batch, n_iter: quante partite vogliamo giocare per il test (più partite giochiamo più è preciso)
rand_ratio: proporzione partite giocate dal secondo
second-player giocatore utilizzato come riferimento per misurare la forza

rand_ratio = 0.2 -> 20% delle mosse le gioca second-player (unicamente per il secondo giocatore)

1a mossa: player VS second-player
2a mossa: player VS player
3a mossa: player VS player
4a mossa: player VS player
5a mossa: player VS second-player
6a mossa: player VS player
7a mossa: player VS player
8a mossa: player VS player
9a mossa: player VS player
10a mossa: player VS player
11a mossa: player VS player
12a mossa: player VS second-player
13a mossa: player VS player
14a mossa: player VS player

se player e second-player hanno la stessa forza allora vinceranno lo stesso numero di partite score = (ratio_vinte - ratio_perse) = 0

se player è un po' più forte di second player allora ratio_vinte > ratio_perse (di poco)

se player è molto molto più forte di second-player allora ratio_vinte >> ratio_perse  $\simeq$ 1

Analogia con gli scacchi: prendi due bambini (player) e rimpiazzi qualcuna delle loro mosse con delle mosse a caso: cambia poco.
Prendo due gran maestri (player) e rimpiazzo alcune delle mosse di uno di loro con mosse a caso: quasi sicuramente il giocatore al quale rimiazzo le mosse perderà. 
Questo perché il livello della partita è più alto e ogni sbaglio ha più peso

Due modi di vederlo:
- Più il livello è alto, più gli sbagli pesano
- Più due livelli sono diversi, più una mescolanza dei due avrà forza diversa rispetto al giocatore originale (es: mescolo birra con acqua, oppure grappa con acqua)

**Possiamo misurare la forza di un giocatore in modo autoreferenziale e trovare un indicatore che permette di confrontare la forza di due giocatori, senza bisogno che si affrontino**
(Aggiungere grafico)


```python
def mirror_score( player: AIPlayer, 
                  nbatch        = 1, 
                  n_iter        = 200, 
                  rand_ratio    = 0.2, 
                  second_player = None, 
                  cols          = 7, 
                  device        = torch.device("cpu")):
    
    """Make the model play against a randomized version of itself"""
    batch_board = BatchBoard(nbatch=nbatch, device=device)
    n_match = win = lost = error = 0
    if second_player == None:
        
        # Default second-player is the greedy player    
        second_player = GreedyModel()
        
    for i in range(n_iter):
        
        # Not randomized player
        move     = player.play(batch_board)
        summary  = batch_board.get_reward(move)
        n_match += summary["is_final"].sum().item()
        win     += summary["has_win"].sum().item()
        error   += (~summary["is_valid"]).sum().item()

        # Randomized player
        move = player.play(batch_board)
        
        # Choose a random move or a greedy move with rand_ratio probability
        #rand_move = torch.randint(0, cols, [nbatch], device=device)
        second_move = second_player.play(batch_board)
        
        # When choosing a random move (or a greedy move)
        rand_choice = torch.rand([nbatch], device=device)

        move[rand_choice < rand_ratio] = second_move[rand_choice < rand_ratio]
        
        # play the move
        summary     = batch_board.get_reward(move)
        n_match    += summary["is_final"].sum().item()
        lost       += summary["has_win"].sum().item()
        tot_summary = {
                        "n_match"     : n_match,
                        "average_len" : (nbatch * n_iter) / n_match,
                        "ratio_error" : error / n_match,
                        "ratio_win"   : win / n_match,
                        "ratio_lost"  : lost / n_match,
                        #"score": win / (lost + error)
                        "score"       : (win - lost - 3 * error) / n_match
                       }
        
    return tot_summary
```

   
# 5. (in sospeso) Confronto tra modelli

   a) ledear board

   b) validazione mirror score

# 6. (Rino) Passare un modello in produzione (javascript)

[https://codepen.io/osbulbul/pen/ngJdYy](https://)

# 7. pubblicità contest (sotto i 500K)

