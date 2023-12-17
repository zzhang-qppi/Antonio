import copy

from chesssss import Board, result, is_in_check
from fentoimage.board import BoardImage

f = '8/5R2/5k2/8/rN6/6K1/8/8 w - - 0 1'

renderer = BoardImage(fen=f)
image = renderer.render()
#image.show()

b = Board(fen=f)
print(b.check_terminal())
print(is_in_check(b, False))
print(result(b))