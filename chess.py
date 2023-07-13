import numpy as np
class Piece():
    def __init__(self, color, location):
        # c is a boolean. True means white. False means black.
        # l is a tuple that represents the indices of the piece in board.
        self.color = color
        self.location = location

    def all_moves(self, b):  # b is a Board type
        return []

    def __eq__(self, other):
        return type(self) == type(other) and self.color == other.color and self.location == other.location

class Pawn(Piece):
    def all_moves(self, b):
        m = []
        r, f = self.location
        board = b.board
        if self.color:
            try:
                if type(board[r+1][f-1]) == Piece and not board[r+1][f-1].color:
                    p = np.copy(b)
                    p[r][f] = None
                    p[r+1][f-1] = Pawn(self.color, (r+1, f-1))
                    m.append(p)
            except IndexError:
                pass
            try:
                if type(board[r+1][f+1]) == Piece and not board[r+1][f+1].color:
                    p = np.copy(b)
                    p[r][f] = None
                    p[r+1][f+1] = Pawn(self.color, (r+1, f+1))
                    m.append(p)
            except IndexError:
                pass
        else:
            try:
                if type(board[r-1][f-1]) == Piece and not board[r-1][f-1].color:
                    p = np.copy(b)
                    p[r][f] = None
                    p[r-1][f-1] = Pawn(self.color, (r-1, f-1))
                    m.append(p)
            except IndexError:
                pass
            try:
                if type(board[r-1][f+1]) == Piece and not board[r-1][f+1].color:
                    p = np.copy(b)
                    p[r][f] = None
                    p[r-1][f+1] = Pawn(self.color, (r-1, f+1))
                    m.append(p)
            except IndexError:
                pass

        return np.array(m)

class Rook(Piece):
    def all_moves(self, b):
        m = []
        r, f = self.location
        board = b.board
        for i in reversed(range(r)):
            if type(board[i][f]) == Piece and board[i][f].color is self.color:
                break
            else:
                p = b.copy()
                p.board[i][f] = Rook(self.color, (i,f))
                p.board[r][f] = None
                m.append(p)
        for i in range(r+1, 8):
            if type(board[i][f]) == Piece and board[i][f].color is self.color:
                break
            else:
                p = b.copy()
                p.board[i][f] = Rook(self.color, (i,f))
                p.board[r][f] = None
                m.append(p)
        for i in reversed(range(f)):
            if type(board[r][i]) == Piece and board[r][i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r][i] = Rook(self.color, (r, i))
                p.board[r][f] = None
                m.append(p)
        for i in range(r+1, 8):
            if type(board[r][i]) == Piece and board[r][i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r][i] = Rook(self.color, (r, i))
                p.board[r][f] = None
                m.append(p)
        return np.array(m)

class Bishop(Piece):
    def all_moves(self, b):
        m = []
        return m

class Knight(Piece):
    def all_moves(self, b):
        m = []
        return m

class Queen(Piece):
    def all_moves(self, b):
        m = []
        return m

class King(Piece):
    def all_moves(board):
        m = []
        return m

class Board():
    def __init__(self, fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'):
        flist = fen.split()
        self.fen = fen
        self.board = convert_board(convert_fen(flist[0]))
        self.turn = flist[1]
        self.castling_rights = flist[2]
        self.enpassant = flist[3]
        self.possible_moves = []
        self.is_terminal = is_terminal(self)
        if int(flist[4]) >= 100:
            self.is_terminal = True

    def expand(self):
        self.possible_moves = move_generation(self, self.turn)

    def copy(self):
        return Board(self.fen)

def convert_fen(f):
    board = np.array([[None for j in range(8)] for i in range(8)])  # board[rank, file]
    ranks = f.split("/")  # a list of strings, each of which represents a rank
    for rank in ranks:
        pointer = 0
        for i in rank:
            if pointer == 8:
                break
            if i in [f"{j+1}" for j in range(8)]:
                pointer += int(i)
            else:
                board[rank, pointer] = i
                pointer += 1
    return board

def convert_board(board):
    # board is an 8*8 array filled with characters from FEN. convertBoard() replaces the characters with Piece objects.
    white_pieces = []
    black_pieces = []
    for rank in board:
        for file in rank:
            current_charact = board[rank, file]
            match current_charact:
                case "p":
                    current_piece = Pawn(False, (rank, file))
                    board[rank, file] = current_piece
                    black_pieces[current_charact] = current_piece
                case "r":
                    current_piece = Rook(False, (rank, file))
                    board[rank, file] = current_piece
                    black_pieces[current_charact] = current_piece
                case "n":
                    current_piece = Knight(False, (rank, file))
                    board[rank, file] = current_piece
                    black_pieces[current_charact] = current_piece
                case "b":
                    current_piece = Bishop(False, (rank, file))
                    board[rank, file] = current_piece
                    black_pieces[current_charact] = current_piece
                case "q":
                    current_piece = Queen(False, (rank, file))
                    board[rank, file] = current_piece
                    black_pieces[current_charact] = current_piece
                case "k":
                    current_piece = King(False, (rank, file))
                    board[rank, file] = current_piece
                    black_pieces[current_charact] = current_piece
                case "P":
                    current_piece = Pawn(True, (rank, file))
                    board[rank, file] = current_piece
                    white_pieces[current_charact] = current_piece
                case "R":
                    current_piece = Rook(True, (rank, file))
                    board[rank, file] = current_piece
                    white_pieces[current_charact] = current_piece
                case "N":
                    current_piece = Knight(True, (rank, file))
                    board[rank, file] = current_piece
                    white_pieces[current_charact] = current_piece
                case "B":
                    current_piece = Bishop(True, (rank, file))
                    board[rank, file] = current_piece
                    white_pieces[current_charact] = current_piece
                case "Q":
                    current_piece = Queen(True, (rank, file))
                    board[rank, file] = current_piece
                    white_pieces[current_charact] = current_piece
                case "K":
                    current_piece = King(True, (rank, file))
                    board[rank, file] = current_piece
                    white_pieces[current_charact] = current_piece
    return board, white_pieces, black_pieces

def is_in_check(b, active_color):
    # return True if the king is in check; False otherwise

    board = b.board
    for x in range(8):
        for y in range(8):
            if board[x][y] == King(active_color, (x, y)):
                king_rank = x
                king_file = y

    # pawn check
    if king_rank < 7 and (
            board[king_rank+1, king_file+1] == Pawn(not active_color, (king_rank+1, king_file+1)) or (
            board[king_rank+1, king_file-1] == Pawn(not active_color, (king_rank+1, king_file-1)))):
        return True

    # knight check
    knight_positions = [(king_rank + 1, king_file + 2),
                        (king_rank + 1, king_file - 2),
                        (king_rank - 1, king_file + 2),
                        (king_rank - 1, king_file - 2),
                        (king_rank + 2, king_file + 1),
                        (king_rank + 2, king_file - 1),
                        (king_rank - 2, king_file + 1),
                        (king_rank - 2, king_file - 1)]
    for p in knight_positions:
        try:
            n = board[p]
            if n == Knight(not active_color, p):
                return True
        except IndexError:
            continue

    # bishop check and diagonal queen check

#    for i in range(1, min(8-king_rank, 8-king_file)):  #up-right
#        if board[king_rank+i, king_file+i] == Bishop(not active_color, [king_rank+i, king_file+i]):
#            return True
#        if board[king_rank+i, king_file+i] == Bishop(active_color, [king_rank+i, king_file+i]):
#            break
#    for i in range(1, min(8-king_rank, king_file)):  #up-left
#        if board[king_rank+i, king_file-i] == Bishop(not active_color, [king_rank+i, king_file-i]):
#            return True
#        if board[king_rank+i, king_file-i] == Bishop(active_color, [king_rank+i, king_file-i]):
#            break
#    for i in range(1, min(king_rank, king_file)):  #down-left
#        if board[king_rank-i, king_file-i] == Bishop(not active_color, [king_rank-i, king_file-i]):
#            return True
#        if board[king_rank-i, king_file-i] == Bishop(active_color, [king_rank-i, king_file-i]):
#            break
#    for i in range(1,min(king_rank,8-king_file)):  #down-right

    for i in range(8):
        try:
            if board[king_rank+i, king_file+i] == Bishop(not active_color, (king_rank+i, king_file+i)):
                return True
            if board[king_rank+i, king_file+i] == Queen(not active_color, (king_rank+i, king_file+i)):
                return True
            if board[king_rank+i, king_file+i] != None:
                break
        except IndexError:
            break
    for i in range(8):
        try:
            if board[king_rank+i, king_file-i] == Bishop(not active_color, (king_rank+i, king_file-i)):
                return True
            if board[king_rank+i, king_file-i] == Queen(not active_color, (king_rank+i, king_file-i)):
                return True
            if board[king_rank+i, king_file-i] != None:
                break
        except IndexError:
            break
    for i in range(8):
        try:
            if board[king_rank-i, king_file-i] == Bishop(not active_color, (king_rank-i, king_file-i)):
                return True
            if board[king_rank-i, king_file-i] == Queen(not active_color, (king_rank-i, king_file-i)):
                return True
            if board[king_rank-i, king_file-i] != None:
                break
        except IndexError:
            break
    for i in range(8):
        try:
            if board[king_rank-i, king_file+i] == Bishop(not active_color, (king_rank-i, king_file+i)):
                return True
            if board[king_rank-i, king_file+i] == Queen(not active_color, (king_rank-i, king_file+i)):
                return True
            if board[king_rank-i, king_file+i] != None:
                break
        except IndexError:
            break

    # rook check and linear queen check
    for i in range(8):
        try:
            if board[king_rank+i, king_file] == Rook(not active_color, (king_rank+i, king_file)):
                return True
            if board[king_rank+i, king_file] == Queen(not active_color, (king_rank+i, king_file)):
                return True
            if board[king_rank+i, king_file] != None:
                break
        except IndexError:
            break
    for i in range(8):
        try:
            if board[king_rank-i, king_file] == Rook(not active_color, (king_rank-i, king_file)):
                return True
            if board[king_rank-i, king_file] == Queen(not active_color, (king_rank-i, king_file)):
                return True
            if board[king_rank-i, king_file] != None:
                break
        except IndexError:
            break
    for i in range(8):
        try:
            if board[king_rank, king_file+i] == Rook(not active_color, (king_rank, king_file+1)):
                return True
            if board[king_rank, king_file+i] == Queen(not active_color, (king_rank, king_file+1)):
                return True
            if board[king_rank, king_file+i] != None:
                break
        except IndexError:
            break
    for i in range(8):
        try:
            if board[king_rank, king_file-i] == Rook(not active_color, (king_rank, king_file-1)):
                return True
            if board[king_rank, king_file-i] == Queen(not active_color, (king_rank, king_file-1)):
                return True
            if board[king_rank, king_file-i] != None:
                break
        except IndexError:
            break

    # king check
    for r in [king_rank, king_rank+1, king_rank-1]:
        for f in [king_file, king_file+1, king_file-1]:
            if board[r, f] == King(not active_color, (r, f)):
                return True

    return False

def move_generation(b, active_color):
    actions = []  # a list of FEN strings
    for square in np.nditer(b.board):
        if square.color is active_color and type(square) == Piece:
            actions.append(square.all_moves)

    for i in len(actions):
        if is_in_check(actions[i], active_color):
            del actions[i]
    return actions

def is_terminal(board):
    return len(move_generation(board)) == 0

def result(board, turn):
    if not is_terminal(board): return 0
    if is_in_check(board, turn): return -1
    if is_in_check(board, not turn): return 1
    return 0.5
