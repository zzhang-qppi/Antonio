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
            if r + 1 < 7 and f - 1 >= 0:
                if type(board[r + 1, f - 1]) == Piece and not board[r + 1, f - 1].color:
                    p = b.copy()
                    p.board[r, f] = None
                    p.board[r + 1, f - 1] = Pawn(self.color, (r + 1, f - 1))
                    p.halfmoves += 1
                    m.append(p)
            if r + 1 < 7 and f + 1 <= 7:
                if type(board[r + 1, f + 1]) == Piece and not board[r + 1, f + 1].color:
                    p = b.copy()
                    p.board[r, f] = None
                    p.board[r + 1, f + 1] = Pawn(self.color, (r + 1, f + 1))
                    p.halfmoves += 1
                    m.append(p)

            # promotion after a capture
            if r + 1 == 7 and f - 1 >= 0:
                if type(board[r + 1, f - 1]) == Piece and not board[r + 1, f - 1].color:
                    for P in (Queen, Knight, Rook, Bishop):
                        p = b.copy()
                        p.board[r, f] = None
                        p.board[r + 1, f - 1] = P(self.color, (r + 1, f - 1))
                        p.halfmoves += 1
                        m.append(p)
            if r + 1 == 7 and f + 1 <= 7:
                if type(board[r + 1, f + 1]) == Piece and not board[r + 1, f + 1].color:
                    for P in (Queen, Knight, Rook, Bishop):
                        p = b.copy()
                        p.board[r, f] = None
                        p.board[r + 1, f + 1] = P(self.color, (r + 1, f + 1))
                        p.halfmoves += 1
                        m.append(p)

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
            if b.enpassant != -1:
                if b.enpassant + 1 <= 7 and type(board[3, b.enpassant + 1]) == Pawn and board[
                    3, b.enpassant + 1].color == self.color:
                    p = b.copy()
                    p.board[3, b.enpassant + 1] = None
                    p.board[3, b.enpassant] = None
                    p.board[3 + 1, b.enpassant] = Pawn(self.color, (3 + 1, b.enpassant))
                    p.halfmoves += 1
                    m.append(p)
                if b.enpassant - 1 >= 0 and type(board[3, b.enpassant - 1]) == Pawn and board[
                    3, b.enpassant - 1].color == self.color:
                    p = b.copy()
                    p.board[3, b.enpassant - 1] = None
                    p.board[3, b.enpassant] = None
                    p.board[3 + 1, b.enpassant] = Pawn(self.color, (3 + 1, b.enpassant))
                    p.halfmoves += 1
                    m.append(p)

        else:
            # pawn captures
            if r - 1 > 0 and f - 1 >= 0:
                if type(board[r - 1, f - 1]) == Piece and not board[r - 1, f - 1].color:
                    p = b.copy()
                    p.board[r, f] = None
                    p.board[r - 1, f - 1] = Pawn(self.color, (r - 1, f - 1))
                    p.halfmoves += 1
                    m.append(p)
            if r - 1 > 0 and f + 1 <= 7:
                if type(board[r - 1, f + 1]) == Piece and not board[r - 1, f + 1].color:
                    p = b.copy()
                    p.board[r, f] = None
                    p.board[r - 1, f + 1] = Pawn(self.color, (r - 1, f + 1))
                    p.halfmoves += 1
                    m.append(p)

            # promotion after a capture
            if r - 1 == 0 and f - 1 >= 0:
                if type(board[r - 1, f - 1]) == Piece and not board[r - 1, f - 1].color:
                    for P in (Queen, Knight, Rook, Bishop):
                        p = b.copy()
                        p.board[r, f] = None
                        p.board[r - 1, f - 1] = P(self.color, (r - 1, f - 1))
                        p.halfmoves += 1
                        m.append(p)
            if r - 1 == 0 and f + 1 <= 7:
                if type(board[r - 1, f + 1]) == Piece and not board[r - 1, f + 1].color:
                    for P in (Queen, Knight, Rook, Bishop):
                        p = b.copy()
                        p.board[r, f] = None
                        p.board[r - 1, f + 1] = P(self.color, (r - 1, f + 1))
                        p.halfmoves += 1
                        m.append(p)

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
                if b.enpassant + 1 <= 7 and type(board[4, b.enpassant + 1]) == Pawn and board[
                    4, b.enpassant + 1].color == self.color:
                    p = b.copy()
                    p.board[4, b.enpassant + 1] = None
                    p.board[4, b.enpassant] = None
                    p.board[4 - 1, b.enpassant] = Pawn(self.color, (4 - 1, b.enpassant))
                    p.enpassant = -1
                    p.halfmoves += 1
                    m.append(p)
                if b.enpassant - 1 >= 0 and type(board[4, b.enpassant - 1]) == Pawn and board[
                    4, b.enpassant - 1].color == self.color:
                    p = b.copy()
                    p.board[4, b.enpassant - 1] = None
                    p.board[4, b.enpassant] = None
                    p.board[4 - 1, b.enpassant] = Pawn(self.color, (4 - 1, b.enpassant))
                    p.enpassant = -1
                    p.halfmoves += 1
                    m.append(p)
        return m


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
                if board[i, f] is not None: p.halfmoves += 1
                p.board[i, f] = Rook(self.color, (i, f))
                p.board[r, f] = None
                m.append(p)
        for i in range(r + 1, 8):
            if type(board[i, f]) == Piece and board[i, f].color is self.color:
                break
            else:
                p = b.copy()
                if board[i, f] is not None: p.halfmoves += 1
                p.board[i, f] = Rook(self.color, (i, f))
                p.board[r, f] = None
                m.append(p)
        for i in reversed(range(f)):
            if type(board[r, i]) == Piece and board[r, i].color is self.color:
                break
            else:
                p = b.copy()
                if board[r, i] is not None: p.halfmoves += 1
                p.board[r, i] = Rook(self.color, (r, i))
                p.board[r, f] = None
                m.append(p)
        for i in range(f + 1, 8):
            if type(board[r, i]) == Piece and board[r, i].color is self.color:
                break
            else:
                p = b.copy()
                if board[r, i] is not None: p.halfmoves += 1
                p.board[r, i] = Rook(self.color, (r, i))
                p.board[r, f] = None
                m.append(p)

        # change to castling rights after rook moves
        for move in m:
            if r == 0 and self.color:
                if f == 0:
                    move.castling_rights = move.castling_rights.replace("Q", "")
                elif f == 7:
                    move.castling_rights = move.caslting_rights.replace("K", "")
            elif r == 7 and not self.color:
                if f == 0:
                    move.castling_rights = move.caslting_rights.replace("q", "")
                elif f == 7:
                    move.castling_rights = move.caslting_rights.replace("k", "")
        return m


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
                if board[r + i, f + i] is not None: p.halfmoves += 1
                p.board[r + i, f + i] = Bishop(self.color, (r + i, f + i))
                p.board[r, f] = None
                m.append(p)
        for i in range(1, min(f + 1, 8 - r)):
            if type(board[r + i, f - i]) == Piece and board[r + i, f - i].color is self.color:
                break
            else:
                p = b.copy()
                if board[r + i, f - i] is not None: p.halfmoves += 1
                p.board[r + i, f - i] = Bishop(self.color, (r + i, f - i))
                p.board[r, f] = None
                m.append(p)
        for i in range(1, min(r, f) + 1):
            if type(board[r - i, f - i]) == Piece and board[r - i, f - i].color is self.color:
                break
            else:
                p = b.copy()
                if board[r - i, f - i] is not None: p.halfmoves += 1
                p.board[r - i, f - i] = Bishop(self.color, (r - i, f - i))
                p.board[r, f] = None
                m.append(p)
        for i in range(1, min(r + 1, 8 - f)):
            if type(board[r - i, f + i]) == Piece and board[r - i, f + i].color is self.color:
                break
            else:
                p = b.copy()
                if board[r - i, f + i] is not None: p.halfmoves += 1
                p.board[r - i, f + i] = Bishop(self.color, (r - i, f + i))
                p.board[r, f] = None
                m.append(p)
        return m


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
                    if board[i] is not None: p.halfmoves += 1
                    p.board[i] = Knight(self.color, i)
                    p.board[r, f] = None
                    m.append(p)
        return m


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
                if board[i, f] is not None: p.halfmoves += 1
                p.board[i, f] = Queen(self.color, (i, f))
                p.board[r, f] = None
                m.append(p)
        for i in range(r + 1, 8):
            if type(board[i, f]) == Piece and board[i, f].color is self.color:
                break
            else:
                p = b.copy()
                if board[i, f] is not None: p.halfmoves += 1
                p.board[i, f] = Queen(self.color, (i, f))
                p.board[r, f] = None
                m.append(p)
        for i in reversed(range(f)):
            if type(board[r, i]) == Piece and board[r, i].color is self.color:
                break
            else:
                p = b.copy()
                if board[r, i] is not None: p.halfmoves += 1
                p.board[r, i] = Queen(self.color, (r, i))
                p.board[r, f] = None
                m.append(p)
        for i in range(f + 1, 8):
            if type(board[r, i]) == Piece and board[r, i].color is self.color:
                break
            else:
                p = b.copy()
                if board[r, i] is not None: p.halfmoves += 1
                p.board[r, i] = Queen(self.color, (r, i))
                p.board[r, f] = None
                m.append(p)
        for i in range(1, 8 - max(r, f)):
            if type(board[r + i, f + i]) == Piece and board[r + i, f + i].color is self.color:
                break
            else:
                p = b.copy()
                if board[r + i, f + i] is not None: p.halfmoves += 1
                p.board[r + i, f + i] = Queen(self.color, (r + i, f + i))
                p.board[r, f] = None
                m.append(p)
        for i in range(1, min(f + 1, 8 - r)):
            if type(board[r + i, f - i]) == Piece and board[r + i, f - i].color is self.color:
                break
            else:
                p = b.copy()
                if board[r + i, f - i] is not None: p.halfmoves += 1
                p.board[r + i, f - i] = Queen(self.color, (r + i, f - i))
                p.board[r, f] = None
                m.append(p)
        for i in range(1, min(r, f) + 1):
            if type(board[r - i, f - i]) == Piece and board[r - i, f - i].color is self.color:
                break
            else:
                p = b.copy()
                if board[r - i, f - i] is not None: p.halfmoves += 1
                p.board[r - i, f - i] = Queen(self.color, (r - i, f - i))
                p.board[r, f] = None
                m.append(p)
        for i in range(1, min(r + 1, 8 - f)):
            if type(board[r - i, f + i]) == Piece and board[r - i, f + i].color is self.color:
                break
            else:
                p = b.copy()
                if board[r - i, f + i] is not None: p.halfmoves += 1
                p.board[r - i, f + i] = Queen(self.color, (r - i, f + i))
                p.board[r, f] = None
                m.append(p)
        return m


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
                    if board[i, j] is not None: p.halfmoves += 1
                    p.board[i, j] = King(self.color, (i, j))
                    p.board[r, f] = None
                    if self.color and ("K" in p.castling_rights or "Q" in p.castling_rights):
                        p.castling_rights = p.castling_rights.replace("K", "")
                        p.castling_rights = p.castling_rights.replace("Q", "")
                    elif not self.color and ("k" in p.castling_rights or "q" in p.castling_rights):
                        p.castling_rights = p.castling_rights.replace("k", "")
                        p.castling_rights = p.castling_rights.replace("q", "")
                    m.append(p)
        return m


class Board:
    def __init__(self, fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
        flist = fen.split()
        self.fen = fen
        self.board = convert_board(convert_fen(flist[0]))
        self.turn = True if flist[1] == "w" else False
        self.castling_rights = flist[2]
        self.enpassant = -1 if flist[3] == "-" else ord(flist[3][0]) - 96
        self.possible_moves = []
        self.is_expanded = False
        self.halfmoves = int(flist[4])
        self.fullmoves = int(flist[5])
        self.is_terminal = self.halfmoves == 50 or is_terminal(self)
        if int(flist[4]) >= 100:
            self.is_terminal = True

    def expand(self):
        if self.is_expanded:
            return
        else:
            self.possible_moves = move_generation(self)
            self.is_expanded = True
        return

    def copy(self):
        a = Board(self.fen)
        return a

    def to_bitboard(self):
        P, R, N, B, Q, K, p, r, n, self, q, k = np.zeros([12, 8, 8], dtype=np.int8)
        for i in range(8):
            for j in range(8):
                d = self.board[i, j]
                if d is None:
                    continue
                elif type(d) == Pawn:
                    if d.color:
                        P[i, j] = 1
                    else:
                        p[i, j] = 1
                elif type(d) == Rook:
                    if d.color:
                        R[i, j] = 1
                    else:
                        r[i, j] = 1
                elif type(d) == Knight:
                    if d.color:
                        N[i, j] = 1
                    else:
                        n[i, j] = 1
                elif type(d) == Bishop:
                    if d.color:
                        B[i, j] = 1
                    else:
                        self[i, j] = 1
                elif type(d) == Queen:
                    if d.color:
                        Q[i, j] = 1
                    else:
                        q[i, j] = 1
                elif type(d) == King:
                    if d.color:
                        K[i, j] = 1
                    else:
                        k[i, j] = 1
        for arr in [P, R, N, B, Q, K, p, r, n, self, q, k]:
            arr = arr.flatten(order='F')
        piece_placement = np.concatenate([P, R, N, B, Q, K, p, r, n, self, q, k], dtype=np.int8)
        extra_info = np.zeros(1 + 4 + 8 + 1, dtype=np.int8)
        # active color: white=1, black=0
        extra_info[0] = 1 if self.turn else 0

        # castling rights
        if "K" in self.castling_rights: extra_info[1] = 1
        if "Q" in self.castling_rights: extra_info[2] = 1
        if "k" in self.castling_rights: extra_info[3] = 1
        if "q" in self.castling_rights: extra_info[4] = 1

        # enpassant
        extra_info[5 + self.enpassant] = 1 * (self.enpassant != -1)

        # halfmove
        extra_info[13] = self.halfmoves

        return np.concatenate((piece_placement, extra_info), dtype=np.int8)


def convert_fen(f):
    board = [[None for j in range(8)] for i in range(8)]  # board[rank, file]
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


def is_in_check(b):
    # return True if the king is in check; False otherwise

    board = b.board
    active_color = not b.turn  # the side that would be in check is the opposite of the active color next turn
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
    P, R, N, B, Q, K, p, r, n, b, q, k = np.zeros([12, 8, 8], dtype=np.int8)
    for i in range(8):
        for j in range(8):
            d = b.board[i, j]
            if d is None: continue
            elif type(d) == Pawn:
                if d.color: P[i, j] = 1
                else: p[i, j] = 1
            elif type(d) == Rook:
                if d.color: R[i, j] = 1
                else: r[i, j] = 1
            elif type(d) == Knight:
                if d.color: N[i, j] = 1
                else: n[i, j] = 1
            elif type(d) == Bishop:
                if d.color: B[i, j] = 1
                else: b[i, j] = 1
            elif type(d) == Queen:
                if d.color: Q[i, j] = 1
                else: q[i, j] = 1
            elif type(d) == King:
                if d.color: K[i, j] = 1
                else: k[i, j] = 1
    for arr in [P, R, N, B, Q, K, p, r, n, b, q, k]:
        arr = arr.flatten(order='F')
    piece_placement = np.concatenate([P, R, N, B, Q, K, p, r, n, b, q, k], dtype=np.int8)
    extra_info = np.zeros(1+4+8+1, dtype=np.int8)
    # active color: white=1, black=0
    extra_info[0] = 1 if b.turn else 0

    # castling rights
    if "K" in b.castling_rights: extra_info[1] = 1
    if "Q" in b.castling_rights: extra_info[2] = 1
    if "k" in b.castling_rights: extra_info[3] = 1
    if "q" in b.castling_rights: extra_info[4] = 1

    # enpassant
    extra_info[5 + b.enpassant] = 1 * (b.enpassant != -1)

    # halfmove
    extra_info[13] = b.halfmoves

    return np.concatenate((piece_placement, extra_info), dtype=np.int8)


def move_generation(b):
    if is_terminal(b): return -1
    actions = []  # a list of FEN strings
    for r in b.board:
        for square in r:
            if (type(square) == Piece) and (square.color is b.turn):
                actions.append(square.all_moves)

    for i in len(actions):
        if is_in_check(actions[i]):
            del actions[i]
        else:
            if actions[i].turn is False: actions[i].fullmoves += 1
            actions[i].turn = not actions[i].turn
            actions[i].enpassant = -1
    return actions


def is_terminal(b):
    b.expand()
    return len(b.possible_moves) == 0


def result(b, turn):
    if not is_terminal(b): return 0
    if is_in_check(b, turn): return -1
    if is_in_check(b, not turn): return 1
    return 0.5
