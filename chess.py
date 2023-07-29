import numpy as np


class Piece:
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
            # pawn captures
            if r + 1 <= 7 and f - 1 >= 0:
                if type(board[r + 1, f - 1]) == Piece and not board[r + 1, f - 1].color:
                    p = b.copy()
                    p.board[r, f] = None
                    p.board[r + 1, f - 1] = Pawn(self.color, (r + 1, f - 1))
                    p.halfmoves += 1
                    m.append(p)
            if r + 1 <= 7 and f + 1 <= 7:
                if type(board[r + 1, f + 1]) == Piece and not board[r + 1, f + 1].color:
                    p = b.copy()
                    p.board[r, f] = None
                    p.board[r + 1, f + 1] = Pawn(self.color, (r + 1, f + 1))
                    p.halfmoves += 1
                    m.append(p)

                # promotion after a capture


            # pawn advancements
            if r + 1 <= 6 and board[r + 1, f] is None:
                p = b.copy()
                p.board[r, f] = None
                p.board[r + 1, f] = Pawn(self.color, (r + 1, f))
                p.halfmoves += 1
                m.append(p)
            if r == 1 and board[3, f] is None:
                p = b.copy()
                p.board[r, f] = None
                p.board[3, f] = Pawn(self.color, (3, f))
                p.halfmoves += 1
                p.enpassant = (3, f)
                m.append(p)
            # pawn promotion
            if r == 6 and board[7, f] is None:
                for Prom in [Queen, Rook, Bishop, Knight]:
                    p = b.copy()
                    p.board[6, f] = None
                    p.board[7, f] = Prom(self.color, (7, f))
                    p.halfmoves += 1
                    m.append(p)
            # en passant
            if b.enpassant != 0:
                if b.enpassant[1] + 1 <= 7 and type(board[b.enpassant[0], b.enpassant[1] + 1]) == Pawn and board[
                    b.enpassant[0], b.enpassant[1] + 1].color == self.color:
                    p = b.copy()
                    p.board[b.enpassant[0], b.enpassant[1] + 1] = None
                    p.board[b.enpassant[0], b.enpassant[1]] = None
                    p.board[b.enpassant[0] + 1, b.enpassant[1]] = Pawn(self.color, (b.enpassant[0] + 1, b.enpassant[1]))
                    m.append(p)
                if b.enpassant[1] - 1 >= 0 and type(board[b.enpassant[0], b.enpassant[1] - 1]) == Pawn and board[
                    b.enpassant[0], b.enpassant[1] - 1].color == self.color:
                    p = b.copy()
                    p.board[b.enpassant[0], b.enpassant[1] - 1] = None
                    p.board[b.enpassant[0], b.enpassant[1]] = None
                    p.board[b.enpassant[0] + 1, b.enpassant[1]] = Pawn(self.color, (b.enpassant[0] + 1, b.enpassant[1]))
                    m.append(p)

        else:
            # pawn captures
            if r - 1 >= 0 and f - 1 >= 0:
                if type(board[r - 1, f - 1]) == Piece and not board[r - 1, f - 1].color:
                    p = b.copy()
                    p.board[r, f] = None
                    p.board[r - 1, f - 1] = Pawn(self.color, (r - 1, f - 1))
                    m.append(p)
            if r - 1 >= 0 and f + 1 <= 7:
                if type(board[r - 1, f + 1]) == Piece and not board[r - 1, f + 1].color:
                    p = b.copy()
                    p.board[r, f] = None
                    p.board[r - 1, f + 1] = Pawn(self.color, (r - 1, f + 1))
                    m.append(p)

                    # promotion after a capture


            # pawn advancements
            if r - 1 >= 1:
                if board[r - 1, f] is None:
                    p = b.copy()
                    p.board[r, f] = None
                    p.board[r - 1, f] = Pawn(self.color, (r - 1, f))
                    p.halfmoves += 1
                    m.append(p)
            if r == 6 and board[4, f] is None:
                p = b.copy()
                p.board[r, f] = None
                p.board[4, f] = Pawn(self.color, (4, f))
                p.halfmoves += 1
                p.enpassant = (3, f)
                m.append(p)
            # pawn promotion
            if r == 1 and board[0, f] is None:
                for Prom in [Queen, Rook, Bishop, Knight]:
                    p = b.copy()
                    p.board[1, f] = None
                    p.board[0, f] = Prom(self.color, (0, f))
                    p.halfmoves += 1
                    m.append(p)
            # en passant
            if b.enpassant != 0:
                if b.enpassant[1] + 1 <= 7 and type(board[b.enpassant[0], b.enpassant[1] + 1]) == Pawn and board[
                    b.enpassant[0], b.enpassant[1] + 1].color == self.color:
                    p = b.copy()
                    p.board[b.enpassant[0], b.enpassant[1] + 1] = None
                    p.board[b.enpassant[0], b.enpassant[1]] = None
                    p.board[b.enpassant[0] - 1, b.enpassant[1]] = Pawn(self.color, (b.enpassant[0] - 1, b.enpassant[1]))
                    p.enpassant = "-"
                    m.append(p)
                if b.enpassant[1] - 1 >= 0 and type(board[b.enpassant[0], b.enpassant[1] - 1]) == Pawn and board[
                    b.enpassant[0], b.enpassant[1] - 1].color == self.color:
                    p = b.copy()
                    p.board[b.enpassant[0], b.enpassant[1] - 1] = None
                    p.board[b.enpassant[0], b.enpassant[1]] = None
                    p.board[b.enpassant[0] - 1, b.enpassant[1]] = Pawn(self.color, (b.enpassant[0] - 1, b.enpassant[1]))
                    p.enpassant = "-"
                    m.append(p)
        return np.array(m)


class Rook(Piece):
    def all_moves(self, b):
        m = []
        r, f = self.location
        board = b.board

        # change to the castling rights

        for i in reversed(range(r)):
            if type(board[i, f]) == Piece and board[i, f].color is self.color:
                break
            else:
                p = b.copy()
                p.board[i, f] = Rook(self.color, (i, f))
                p.board[r, f] = None
                m.append(p)
        for i in range(r + 1, 8):
            if type(board[i, f]) == Piece and board[i, f].color is self.color:
                break
            else:
                p = b.copy()
                p.board[i, f] = Rook(self.color, (i, f))
                p.board[r, f] = None
                m.append(p)
        for i in reversed(range(f)):
            if type(board[r, i]) == Piece and board[r, i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r, i] = Rook(self.color, (r, i))
                p.board[r, f] = None
                m.append(p)
        for i in range(f + 1, 8):
            if type(board[r, i]) == Piece and board[r, i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r, i] = Rook(self.color, (r, i))
                p.board[r, f] = None
                m.append(p)
        return np.array(m)


class Bishop(Piece):
    def all_moves(self, b):
        m = []
        r, f = self.location
        board = b.board
        for i in range(1, 8 - max(r, f)):
            if type(board[r + i, f + i]) == Piece and board[r + i, f + i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r + i, f + i] = Bishop(self.color, (r + i, f + i))
                p.board[r, f] = None
                m.append(p)
        for i in range(1, min(f + 1, 8 - r)):
            if type(board[r + i, f - i]) == Piece and board[r + i, f - i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r + i, f - i] = Bishop(self.color, (r + i, f - i))
                p.board[r, f] = None
                m.append(p)
        for i in range(1, min(r, f) + 1):
            if type(board[r - i, f - i]) == Piece and board[r - i, f - i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r - i, f - i] = Bishop(self.color, (r - i, f - i))
                p.board[r, f] = None
                m.append(p)
        for i in range(1, min(r + 1, 8 - f)):
            if type(board[r - i, f + i]) == Piece and board[r - i, f + i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r - i, f + i] = Bishop(self.color, (r - i, f + i))
                p.board[r, f] = None
                m.append(p)
        return np.array(m)


class Knight(Piece):
    def all_moves(self, b):
        m = []
        r, f = self.location
        board = b.board
        knight_positions = [(r + 1, f + 2),
                            (r + 1, f - 2),
                            (r - 1, f + 2),
                            (r - 1, f - 2),
                            (r + 2, f + 1),
                            (r + 2, f - 1),
                            (r - 2, f + 1),
                            (r - 2, f - 1)]
        for i in knight_positions:
            if (0 <= i[0] <= 7) and (0 <= i[1] <= 7):
                if not ((type(board[i]) == Piece) and (board[i].color is self.color)):
                    p = b.copy()
                    p.board[i] = Knight(self.color, i)
                    p.board[r, f] = None
                    m.append(p)
        return np.array(m)


class Queen(Piece):
    def all_moves(self, b):
        m = []
        r, f = self.location
        board = b.board
        for i in reversed(range(r)):
            if type(board[i, f]) == Piece and board[i, f].color is self.color:
                break
            else:
                p = b.copy()
                p.board[i, f] = Queen(self.color, (i, f))
                p.board[r, f] = None
                m.append(p)
        for i in range(r + 1, 8):
            if type(board[i, f]) == Piece and board[i, f].color is self.color:
                break
            else:
                p = b.copy()
                p.board[i, f] = Queen(self.color, (i, f))
                p.board[r, f] = None
                m.append(p)
        for i in reversed(range(f)):
            if type(board[r, i]) == Piece and board[r, i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r, i] = Queen(self.color, (r, i))
                p.board[r, f] = None
                m.append(p)
        for i in range(f + 1, 8):
            if type(board[r, i]) == Piece and board[r, i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r, i] = Queen(self.color, (r, i))
                p.board[r, f] = None
                m.append(p)
        for i in range(1, 8 - max(r, f)):
            if type(board[r + i, f + i]) == Piece and board[r + i, f + i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r + i, f + i] = Queen(self.color, (r + i, f + i))
                p.board[r, f] = None
                m.append(p)
        for i in range(1, min(f + 1, 8 - r)):
            if type(board[r + i, f - i]) == Piece and board[r + i, f - i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r + i, f - i] = Queen(self.color, (r + i, f - i))
                p.board[r, f] = None
                m.append(p)
        for i in range(1, min(r, f) + 1):
            if type(board[r - i, f - i]) == Piece and board[r - i, f - i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r - i, f - i] = Queen(self.color, (r - i, f - i))
                p.board[r, f] = None
                m.append(p)
        for i in range(1, min(r + 1, 8 - f)):
            if type(board[r - i, f + i]) == Piece and board[r - i, f + i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r - i, f + i] = Queen(self.color, (r - i, f + i))
                p.board[r, f] = None
                m.append(p)
        return np.array(m)


class King(Piece):
    def all_moves(self, b):
        m = []
        r, f = self.location
        board = b.board
        for i in [r, r + 1, r - 1]:
            for j in [f, f + 1, f - 1]:
                if (0 <= i <= 7) and (0 <= j <= 7) and not (
                        type(board[i, j]) == Piece and board[i, j].color is self.color):
                    p = b.copy()
                    p.board[i, j] = King(self.color, (i, j))
                    p.board[r, f] = None
                    if self.color and ("K" in p.castling_rights or "Q" in p.castling_rights):
                        p.castling_rights = p.castling_rights.replace("K", "")
                        p.castling_rights = p.castling_rights.replace("Q", "")
                    elif not self.color and ("k" in p.castling_rights or "q" in p.castling_rights):
                        p.castling_rights = p.castling_rights.replace("k", "")
                        p.castling_rights = p.castling_rights.replace("q", "")
                    m.append(p)
        return np.array(m)


class Board:
    def __init__(self, fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'):
        flist = fen.split()
        self.fen = fen
        self.board = convert_board(convert_fen(flist[0]))
        self.turn = True if flist[1] == "w" else False
        self.castling_rights = flist[2]
        self.enpassant = 0 if flist[3] == "-" else (ord(flist[3][0]) - 96, int(flist[3][1]))
        self.possible_moves = []
        self.halfmoves = int(flist[4])
        self.fullmoves = int(flist[5])
        self.is_terminal = self.halfmoves == 50 or is_terminal(self)
        if int(flist[4]) >= 100:
            self.is_terminal = True

    def is_expanded(self):
        if is_terminal(self): return -1
        return len(self.possible_moves) != 0

    def expand(self):
        self.possible_moves = move_generation(self, self.turn)
        return 0

    def copy(self):
        a = Board(self.fen)
        a.enpassant = "-"
        return a


def convert_fen(f):
    board = np.array([[None for j in range(8)] for i in range(8)])  # board[rank, file]
    ranks = f.split("/")  # a list of strings, each of which represents a rank
    for rank in ranks:
        pointer = 0
        for i in rank:
            if pointer == 8:
                break
            if i in [f"{j + 1}" for j in range(8)]:
                pointer += int(i)
            else:
                board[rank, pointer] = i
                pointer += 1
    return board


def convert_board(board):
    # board is an 8*8 array filled with characters from FEN. convertBoard() replaces the characters with Piece objects.
    # white_pieces = []
    # black_pieces = []
    for rank in board:
        for file in rank:
            current_charact = board[rank, file]
            match current_charact:
                case "p":
                    current_piece = Pawn(False, (rank, file))
                    board[rank, file] = current_piece
                    # black_pieces[current_charact] = current_piece
                case "r":
                    current_piece = Rook(False, (rank, file))
                    board[rank, file] = current_piece
                    # black_pieces[current_charact] = current_piece
                case "n":
                    current_piece = Knight(False, (rank, file))
                    board[rank, file] = current_piece
                    # black_pieces[current_charact] = current_piece
                case "b":
                    current_piece = Bishop(False, (rank, file))
                    board[rank, file] = current_piece
                    # black_pieces[current_charact] = current_piece
                case "q":
                    current_piece = Queen(False, (rank, file))
                    board[rank, file] = current_piece
                    # black_pieces[current_charact] = current_piece
                case "k":
                    current_piece = King(False, (rank, file))
                    board[rank, file] = current_piece
                    # black_pieces[current_charact] = current_piece
                case "P":
                    current_piece = Pawn(True, (rank, file))
                    board[rank, file] = current_piece
                    # white_pieces[current_charact] = current_piece
                case "R":
                    current_piece = Rook(True, (rank, file))
                    board[rank, file] = current_piece
                    # white_pieces[current_charact] = current_piece
                case "N":
                    current_piece = Knight(True, (rank, file))
                    board[rank, file] = current_piece
                    # white_pieces[current_charact] = current_piece
                case "B":
                    current_piece = Bishop(True, (rank, file))
                    board[rank, file] = current_piece
                    # white_pieces[current_charact] = current_piece
                case "Q":
                    current_piece = Queen(True, (rank, file))
                    board[rank, file] = current_piece
                    # white_pieces[current_charact] = current_piece
                case "K":
                    current_piece = King(True, (rank, file))
                    board[rank, file] = current_piece
                    # white_pieces[current_charact] = current_piece
    return board  # , white_pieces, black_pieces


def is_in_check(b, active_color):
    # return True if the king is in check; False otherwise

    board = b.board
    for x in range(8):
        for y in range(8):
            if board[x, y] == King(active_color, (x, y)):
                king_rank = x
                king_file = y

    # pawn check
    if active_color and king_rank < 6:
        for f in [king_file - 1, king_file + 1]:
            if f >= 0 and f <= 7 and board[king_rank + 1, f] == Pawn(not active_color, (king_rank + 1, f)):
                return True
    if not active_color and king_rank > 1:
        for f in [king_file - 1, king_file + 1]:
            if f >= 0 and f <= 7 and board[king_rank - 1, f] == Pawn(not active_color, (king_rank - 1, f)):
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
        if p[0] >= 0 and p[0] <= 7 and p[1] >= 0 and p[1] <= 7:
            n = board[p]
            if n == Knight(not active_color, p):
                return True

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

    for i in range(1, 8 - max(king_rank, king_file)):
        if board[king_rank + i, king_file + i] == Bishop(not active_color, (king_rank + i, king_file + i)):
            return True
        if board[king_rank + i, king_file + i] == Queen(not active_color, (king_rank + i, king_file + i)):
            return True
        if board[king_rank + i, king_file + i] != None:
            break
    for i in range(1, min(king_file + 1, 8 - king_rank)):
        if board[king_rank + i, king_file - i] == Bishop(not active_color, (king_rank + i, king_file - i)):
            return True
        if board[king_rank + i, king_file - i] == Queen(not active_color, (king_rank + i, king_file - i)):
            return True
        if board[king_rank + i, king_file - i] != None:
            break
    for i in range(1, min(king_rank, king_file) + 1):
        if board[king_rank - i, king_file - i] == Bishop(not active_color, (king_rank - i, king_file - i)):
            return True
        if board[king_rank - i, king_file - i] == Queen(not active_color, (king_rank - i, king_file - i)):
            return True
        if board[king_rank - i, king_file - i] != None:
            break
    for i in range(1, min(king_rank + 1, 8 - king_file)):
        if board[king_rank - i, king_file + i] == Bishop(not active_color, (king_rank - i, king_file + i)):
            return True
        if board[king_rank - i, king_file + i] == Queen(not active_color, (king_rank - i, king_file + i)):
            return True
        if board[king_rank - i, king_file + i] != None:
            break

    # rook check and linear queen check
    for i in range(king_rank + 1, 8):
        if board[i, king_file] == Rook(not active_color, (i, king_file)):
            return True
        if board[i, king_file] == Queen(not active_color, (i, king_file)):
            return True
        if board[i, king_file] != None:
            break
    for i in reversed(range(king_rank)):
        if board[i, king_file] == Rook(not active_color, (i, king_file)):
            return True
        if board[i, king_file] == Queen(not active_color, (i, king_file)):
            return True
        if board[i, king_file] != None:
            break
    for i in range(king_file + 1, 8):
        if board[king_rank, i] == Rook(not active_color, (king_rank, i)):
            return True
        if board[king_rank, i] == Queen(not active_color, (king_rank, i)):
            return True
        if board[king_rank, i] != None:
            break
    for i in reversed(range(king_file)):
        if board[king_rank, i] == Rook(not active_color, (king_rank, i)):
            return True
        if board[king_rank, i] == Queen(not active_color, (king_rank, i)):
            return True
        if board[king_rank, i] != None:
            break

    # king check
    for i in [king_rank, king_rank + 1, king_rank - 1]:
        for j in [king_file, king_file + 1, king_file - 1]:
            if i >= 0 and i <= 7 and j >= 0 and j <= 7 and board[i, j] == King(not active_color, (i, j)):
                return True

    return False


def to_bitboard(b):
    r = []


def move_generation(b, active_color):
    if is_terminal(b): return -1
    actions = []  # a list of FEN strings
    for square in np.nditer(b.board):
        if (type(square) == Piece) and (square.color is active_color):
            actions.append(square.all_moves)

    for i in len(actions):
        if is_in_check(actions[i], active_color):
            del actions[i]
        else:
            if actions[i].turn is False: actions[i].fullmoves += 1
            actions[i].turn = not actions[i].turn
    return np.array(actions)


def is_terminal(b):
    return len(move_generation(b)) == 0


def result(b, turn):
    if not is_terminal(b): return 0
    if is_in_check(b, turn): return -1
    if is_in_check(b, not turn): return 1
    return 0.5
