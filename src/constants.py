from ale_py import Action

POP_SIZE = 1000
PARENT_POP_SIZE = 50
TOURNAMENT_SIZE = 15
AGE_LIMIT = 20
ITER_DEPTH = 100


P_MUTATE = 0.3
ETA = 10

HARD_CODE_PATH = "/Users/eaho/MSci/BigData/venvs/PacBot/lib/python3.10/site-packages/ale_py/roms/ms_pacman.bin"

OUT_MAP = {
    0: Action.NOOP,
    1: Action.UP,
    2: Action.UPRIGHT,
    3: Action.RIGHT,
    4: Action.DOWNRIGHT,
    5: Action.DOWN,
    6: Action.DOWNLEFT,
    7: Action.LEFT,
    8: Action.UPLEFT,
}
