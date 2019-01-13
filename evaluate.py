from game2048.game import Game
from game2048.displays import Display
import random
import time




 

def single_run(size, score_to_win, AgentClass,**kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), **kwargs)
    
    agent.play(verbose=False)
    return game.score


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 50

    '''====================
    Use your own agent here.'''
    from game2048.agents import MyAgent as TestAgent
    
    '''===================='''

    scores = []
    _=0

    start = time.time()


    while(_<N_TESTS):
    # for _ in range(N_TESTS):
        score = single_run(GAME_SIZE, SCORE_TO_WIN,
                           AgentClass=TestAgent)
        scores.append(score)
        print("this time"+str(_)+"score"+str(score))
        _=_+1

    #long running
    end = time.time()
    print(end-start)

    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
