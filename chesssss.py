import numpy as np
import copy
from fentoimage.board import BoardImage
import re


class Piece:
    def __init__(self, color, location):
        '''
        c is a boolean. True means white. False means black.
        l is a tuple that represents the indices of the piece in board. (rank, file)
        '''
        self.color = color
        self.location = location

    def all_moves(self, b):  
        # b is a Board type
        return []

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.color == other.color) and (self.location == other.location)
    
    def __repr__(self):
        return f"{type(self).__name__}/{'w' if self.color else 'b'}"
    
    def __str__(self):
        char = self.__class__.__name__[0]
        if isinstance(self, Knight):
            char = 'n'
        if self.color:
            char = char.upper()
        else:
            char = char.lower()
        return char


class Pawn(Piece):
    def all_moves(self, b):
        m = []
        r, f = self.location
        board = b.board
        if not self.color:
            # pawn captures
            if r + 1 < 7 and f - 1 >= 0:
                if issubclass(type(board[r + 1, f - 1]), Piece) and board[r + 1, f - 1].color is not self.color:
                    p = b.copy()
                    p.board[r, f] = None
                    p.board[r + 1, f - 1] = Pawn(self.color, (r + 1, f - 1))
                    
                    m.append(p)
            if r + 1 < 7 and f + 1 <= 7:
                if issubclass(type(board[r + 1, f + 1]), Piece) and board[r + 1, f + 1].color is not self.color:
                    p = b.copy()
                    p.board[r, f] = None
                    p.board[r + 1, f + 1] = Pawn(self.color, (r + 1, f + 1))
                    
                    m.append(p)

            # promotion after a capture
            if r + 1 == 7 and f - 1 >= 0:
                if issubclass(type(board[r + 1, f - 1]), Piece) and board[r + 1, f - 1].color is not self.color:
                    for P in (Queen, Knight, Rook, Bishop):
                        p = b.copy()
                        p.board[r, f] = None
                        p.board[r + 1, f - 1] = P(self.color, (r + 1, f - 1))
                        m.append(p)
            if r + 1 == 7 and f + 1 <= 7:
                if issubclass(type(board[r + 1, f + 1]), Piece) and board[r + 1, f + 1].color is not self.color:
                    for P in (Queen, Knight, Rook, Bishop):
                        p = b.copy()
                        p.board[r, f] = None
                        p.board[r + 1, f + 1] = P(self.color, (r + 1, f + 1))
                        m.append(p)

            # pawn advancements
            if r + 1 <= 6 and board[r + 1, f] is None:
                p = b.copy()
                p.board[r, f] = None
                p.board[r + 1, f] = Pawn(self.color, (r + 1, f))          
                m.append(p)
            if r == 1 and board[3, f] is None and board[2, f] is None:
                p = b.copy()
                p.board[r, f] = None
                p.board[3, f] = Pawn(self.color, (3, f))
                p.enpassant = (2, f)
                m.append(p)

            # pawn promotion
            if r == 6 and board[7, f] is None:
                for Prom in [Queen, Rook, Bishop, Knight]:
                    p = b.copy()
                    p.board[6, f] = None
                    p.board[7, f] = Prom(self.color, (7, f))
                    m.append(p)

            # en passant
            if b.enpassant is not None:
                if (r, f) == (4, b.enpassant[1]+1):
                    p = b.copy()
                    p.board[r, f] = None
                    p.board[4, f-1] = None
                    p.board[b.enpassant] = Pawn(self.color, b.enpassant)
                    p.enpassant = None
                    m.append(p)
                if (r, f) == (4, b.enpassant[1]-1):
                    p = b.copy()
                    p.board[r, f] = None
                    p.board[4, f+1] = None
                    p.board[b.enpassant] = Pawn(self.color, b.enpassant)
                    p.enpassant = None
                    m.append(p)

        else:
            # pawn captures
            if r - 1 > 0 and f - 1 >= 0:
                if issubclass(type(board[r - 1, f - 1]), Piece) and board[r - 1, f - 1].color is not self.color:
                    p = b.copy()
                    p.board[r, f] = None
                    p.board[r - 1, f - 1] = Pawn(self.color, (r - 1, f - 1))
                    
                    m.append(p)
            if r - 1 > 0 and f + 1 <= 7:
                if issubclass(type(board[r - 1, f + 1]), Piece) and board[r - 1, f + 1].color is not self.color:
                    p = b.copy()
                    p.board[r, f] = None
                    p.board[r - 1, f + 1] = Pawn(self.color, (r - 1, f + 1))
                    
                    m.append(p)

            # promotion after a capture
            if r - 1 == 0 and f - 1 >= 0:
                if issubclass(type(board[r - 1, f - 1]), Piece) and board[r - 1, f - 1].color is not self.color:
                    for P in (Queen, Knight, Rook, Bishop):
                        p = b.copy()
                        p.board[r, f] = None
                        p.board[r - 1, f - 1] = P(self.color, (r - 1, f - 1))
                        
                        m.append(p)
            if r - 1 == 0 and f + 1 <= 7:
                if issubclass(type(board[r - 1, f + 1]), Piece) and board[r - 1, f + 1].color is not self.color:
                    for P in (Queen, Knight, Rook, Bishop):
                        p = b.copy()
                        p.board[r, f] = None
                        p.board[r - 1, f + 1] = P(self.color, (r - 1, f + 1))
                        
                        m.append(p)

            # pawn advancements
            if r - 1 >= 1:
                if board[r - 1, f] is None:
                    p = b.copy()
                    p.board[r, f] = None
                    p.board[r - 1, f] = Pawn(self.color, (r - 1, f))
                    
                    m.append(p)
            if r == 6 and board[4, f] is None and board[5, f] is None:
                p = b.copy()
                p.board[r, f] = None
                p.board[4, f] = Pawn(self.color, (4, f))
                
                p.enpassant = (5, f)
                m.append(p)
                
            # pawn promotion
            if r == 1 and board[0, f] is None:
                for Prom in [Queen, Rook, Bishop, Knight]:
                    p = b.copy()
                    p.board[1, f] = None
                    p.board[0, f] = Prom(self.color, (0, f))
                    
                    m.append(p)
            
            # en passant
            if b.enpassant is not None:
                if (r, f) == (3, b.enpassant[1]+1):
                    p = b.copy()
                    p.board[r, f] = None
                    p.board[3, f-1] = None
                    p.board[b.enpassant] = Pawn(self.color, b.enpassant)
                    m.append(p)
                if (r, f) == (3, b.enpassant[1]-1):
                    p = b.copy()
                    p.board[r, f] = None
                    p.board[3, f+1] = None
                    p.board[b.enpassant] = Pawn(self.color, b.enpassant)
                    m.append(p)
        return m


class Rook(Piece):
    def all_moves(self, b):
        m = []
        r, f = self.location
        board = b.board

        # horizontal move
        for i in reversed(range(r)):
            if issubclass(type(board[i, f]), Piece) and board[i, f].color is self.color:
                break
            else:
                p = b.copy()
                p.board[i, f] = Rook(self.color, (i, f))
                p.board[r, f] = None
                m.append(p)
                if board[i, f] is not None: break
        for i in range(r + 1, 8):
            if issubclass(type(board[i, f]), Piece) and board[i, f].color is self.color:
                break
            else:
                p = b.copy()
                p.board[i, f] = Rook(self.color, (i, f))
                p.board[r, f] = None
                m.append(p)
                if board[i, f] is not None: break
        for i in reversed(range(f)):
            if issubclass(type(board[r, i]), Piece) and board[r, i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r, i] = Rook(self.color, (r, i))
                p.board[r, f] = None
                m.append(p)
                if board[r, i] is not None: break
        for i in range(f + 1, 8):
            if issubclass(type(board[r, i]), Piece) and board[r, i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r, i] = Rook(self.color, (r, i))
                p.board[r, f] = None
                m.append(p)
                if board[r, i] is not None: break

        # change to castling rights after rook moves
        for move in m:
            if r == 7 and self.color:
                if f == 0:
                    move.castling_rights = move.castling_rights.replace("Q", "")
                elif f == 7:
                    move.castling_rights = move.castling_rights.replace("K", "")
            elif r == 0 and not self.color:
                if f == 0:
                    move.castling_rights = move.castling_rights.replace("q", "")
                elif f == 7:
                    move.castling_rights = move.castling_rights.replace("k", "")
        return m


class Bishop(Piece):
    def all_moves(self, b):
        m = []
        r, f = self.location
        board = b.board
        for i in range(1, 8):
            if r + i > 7 or f + i > 7: break
            if issubclass(type(board[r + i, f + i]), Piece) and board[r + i, f + i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r + i, f + i] = Bishop(self.color, (r + i, f + i))
                p.board[r, f] = None
                m.append(p)
                if board[r + i, f + i] is not None: break
        for i in range(1, 8):
            if r + i > 7 or f - i < 0: break
            if issubclass(type(board[r + i, f - i]), Piece) and board[r + i, f - i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r + i, f - i] = Bishop(self.color, (r + i, f - i))
                p.board[r, f] = None
                m.append(p)
                if board[r + i, f - i] is not None: break
        for i in range(1, 8):
            if r - i < 0 or f - i < 0: break
            if issubclass(type(board[r - i, f - i]), Piece) and board[r - i, f - i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r - i, f - i] = Bishop(self.color, (r - i, f - i))
                p.board[r, f] = None
                m.append(p)
                if board[r - i, f - i] is not None: break
        for i in range(1, 8):
            if r - i < 0 or f + i > 7: break
            if issubclass(type(board[r - i, f + i]), Piece) and board[r - i, f + i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r - i, f + i] = Bishop(self.color, (r - i, f + i))
                p.board[r, f] = None
                m.append(p)
                if board[r - i, f + i] is not None: break
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
                if not (issubclass(type(board[i]), Piece) and (board[i].color is self.color)):
                    p = b.copy()
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
            if issubclass(type(board[i, f]), Piece) and board[i, f].color is self.color:
                break
            else:
                p = b.copy()
                p.board[i, f] = Queen(self.color, (i, f))
                p.board[r, f] = None
                m.append(p)
                if board[i, f] is not None: break
        for i in range(r + 1, 8):
            if issubclass(type(board[i, f]), Piece) and board[i, f].color is self.color:
                break
            else:
                p = b.copy()
                p.board[i, f] = Queen(self.color, (i, f))
                p.board[r, f] = None
                m.append(p)
                if board[i, f] is not None: break
        for i in reversed(range(f)):
            if issubclass(type(board[r, i]), Piece) and board[r, i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r, i] = Queen(self.color, (r, i))
                p.board[r, f] = None
                m.append(p)
                if board[r, i] is not None: break
        for i in range(f + 1, 8):
            if issubclass(type(board[r, i]), Piece) and board[r, i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r, i] = Queen(self.color, (r, i))
                p.board[r, f] = None
                m.append(p)
                if board[r, i] is not None: break
        for i in range(1, 8):
            if r + i > 7 or f + i > 7: break
            if issubclass(type(board[r + i, f + i]), Piece) and board[r + i, f + i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r + i, f + i] = Queen(self.color, (r + i, f + i))
                p.board[r, f] = None
                m.append(p)
                if board[r + i, f + i] is not None: break
        for i in range(1, 8):
            if r + i > 7 or f - i < 0: break
            if issubclass(type(board[r + i, f - i]), Piece) and board[r + i, f - i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r + i, f - i] = Queen(self.color, (r + i, f - i))
                p.board[r, f] = None
                m.append(p)
                if board[r + i, f - i] is not None: break
        for i in range(1, 8):
            if r - i < 0 or f - i < 0: break
            if issubclass(type(board[r - i, f - i]), Piece) and board[r - i, f - i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r - i, f - i] = Queen(self.color, (r - i, f - i))
                p.board[r, f] = None
                m.append(p)
                if board[r - i, f - i] is not None: break
        for i in range(1, 8):
            if r - i < 0 or f + i > 7: break
            if issubclass(type(board[r - i, f + i]), Piece) and board[r - i, f + i].color is self.color:
                break
            else:
                p = b.copy()
                p.board[r - i, f + i] = Queen(self.color, (r - i, f + i))
                p.board[r, f] = None
                m.append(p)
                if board[r - i, f + i] is not None: break
        return m


class King(Piece):
    def all_moves(self, b):
        m = []
        r, f = self.location
        board = b.board
        
        # Moves
        for i in [r, r + 1, r - 1]:
            for j in [f, f + 1, f - 1]:
                if (0 <= i <= 7) and (0 <= j <= 7) and not (
                        issubclass(type(board[i, j]), Piece) and board[i, j].color is self.color):
                    p = b.copy()
                    p.board[i, j] = King(self.color, (i, j))
                    p.board[r, f] = None
                    if self.color:
                        p.castling_rights = p.castling_rights.replace("K", "").replace("Q", "")
                    else:
                        p.castling_rights = p.castling_rights.replace("k", "").replace("q", "")
                    m.append(p)
        
        # Castling
        if self.color and 'K' in b.castling_rights:
            if board[7, 5] == board[7, 6] == None:
                p = b.copy()
                p.board[7, 4] = p.board[7, 7] = None
                p.board[7, 6] = King(self.color, (7, 6))
                p.board[7, 5] = Rook(self.color, (7, 5))
                p.castling_rights = p.castling_rights.replace('K','').replace('Q','')
                m.append(p)
        if self.color and 'Q' in b.castling_rights:
            if board[7, 1] == board[7, 2] == board[7, 3] == None:
                p = b.copy()
                p.board[7, 0] = p.board[7, 4] = None
                p.board[7, 2] = King(self.color, (7, 2))
                p.board[7, 3] = Rook(self.color, (7, 3))
                p.castling_rights = p.castling_rights.replace('K','').replace('Q','')
                m.append(p)
        if not self.color and 'k' in b.castling_rights:
            if board[0, 5] == board[0, 6] == None:
                p = b.copy()
                p.board[0, 4] = p.board[7, 7] = None
                p.board[0, 6] = King(self.color, (0, 6))
                p.board[0, 5] = Rook(self.color, (0, 5))
                p.castling_rights = p.castling_rights.replace('k','').replace('q','')
                m.append(p)
        if not self.color and 'q' in b.castling_rights:
            if board[0, 1] == board[0, 2] == board[0, 3] == None:
                p = b.copy()
                p.board[0, 0] = p.board[0, 4] = None
                p.board[0, 2] = King(self.color, (0, 2))
                p.board[0, 3] = Rook(self.color, (0, 3))
                p.castling_rights = p.castling_rights.replace('k','').replace('q','')
                m.append(p)
        return m


class Board:
    def __init__(self, fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
        flist = fen.split()
        self.fen = fen
        self.board = convert_board(convert_fen(flist[0]))
        self.turn = True if flist[1] == "w" else False
        self.castling_rights = flist[2]
        if flist[3] == "-": 
            self.enpassant = None
        else: 
            self.enpassant = (8-int(flist[3][1]), ord(flist[3][0]) - 97)
            assert((self.enpassant[0] == 2 and self.turn) or (self.enpassant[0] == 5 and not self.turn))
        self.possible_moves = []
        self.is_expanded = False
        self.halfmoves = int(flist[4])
        self.fullmoves = int(flist[5])
        self.is_terminal = None

    def expand(self):
        if not self.is_expanded:
            self.possible_moves = np.array(move_generation(self))
            self.is_expanded = True
        return self.possible_moves

    def copy(self):
        a = copy.deepcopy(self)
        a.is_expanded = False
        a.is_terminal = None
        a.fen = ''
        a.possible_moves = []
        return a
    
    def check_terminal(self) -> bool:
        if self.is_terminal is not None:
            return self.is_terminal
        
        if self.halfmoves == 50:
            self.terminal = True
            return True
        
        # insufficient material
        f = self.to_fen().split()[0]
        piece_char_list = re.findall("[a-zA-Z]{1}", f)
        insuff = [set('Kk'), set('KBk'), set('KNk'), set('Kkb'), set('Kkn'), set('KNkn'), set('KBkb'), set('KBkn'), set('KNkb')]
        if set(piece_char_list) in insuff:
            self.is_terminal = True
            return True
            
        if is_mate(self):
            self.terminal = True
            return True
        
        self.terminal = False
        return False

    def to_fen(self) -> str:
        board = self.board
        fen_list = ['']
        for rank in board:
            for square in rank:
                if square is None:
                    if isinstance(fen_list[-1], int):
                        fen_list[-1] += 1
                    else:
                        fen_list.append(1)
                else:
                    fen_list.append(str(square))
            fen_list.append('/')
        fen_list.pop()
        fen_list.append(' w' if self.turn else ' b')
        if self.castling_rights:
            fen_list.append(' ' + self.castling_rights)
        else: fen_list.append(' -')
        if self.enpassant:
            fen_list.append(f" {chr(self.enpassant[1]+97)}{8-self.enpassant[0]}")
        else:
            fen_list.append(' -')
        fen_list.append(f" {self.halfmoves}")
        fen_list.append(f" {self.fullmoves}")
        return ''.join(str(e) for e in fen_list)

    def to_bitboard(self) -> np.ndarray:
        P, R, N, B, Q, K, p, r, n, b, q, k = np.zeros([12, 8, 8])
        # very ugly code here. better change
        for i in range(8):
            for j in range(8):
                d = self.board[i, j]
                if type(d) == Pawn:
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
                        b[i, j] = 1
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
        para = []
        for arr in [P, R, N, B, Q, K, p, r, n, b, q, k]:
            para.append(arr.flatten(order='C'))
        piece_placement = np.concatenate(para)
        extra_info = np.zeros(9, dtype=np.float16)  # [active color, castling rights K, Q, k, q, enpassant[0], enpassant[1], halfmove, fullmove]; len=9
        # active color: white=1, black=0
        extra_info[0] = 1 if self.turn else 0

        # castling rights
        if "K" in self.castling_rights: extra_info[1] = 1
        if "Q" in self.castling_rights: extra_info[2] = 1
        if "k" in self.castling_rights: extra_info[3] = 1
        if "q" in self.castling_rights: extra_info[4] = 1

        # enpassant
        if self.enpassant is not None:
            extra_info[5], extra_info[6] = self.enpassant[0]/7, self.enpassant[1]/7

        # halfmove
        extra_info[7] = self.halfmoves/50
        extra_info[8] = self.fullmoves/200

        return np.concatenate((piece_placement, extra_info), dtype=np.float16)


def convert_fen(f: str) -> np.ndarray:
    board = np.array([[None for j in range(8)] for i in range(8)])  # board\[rank, file\]
    ranks = f.split("/")  # a list of strings, each of which represents a rank
    for r in range(len(ranks)):
        pointer = 0
        for i in ranks[r]:
            if pointer == 8:
                break
            if i in [f"{j + 1}" for j in range(8)]:
                pointer += int(i)
            else:
                board[r, pointer] = i
                pointer += 1
    return board


def convert_board(board: np.ndarray) -> np.ndarray:
    '''
    Replace the characters with Piece objects.
    board is an 8*8 array filled with characters from FEN. 
    '''

    for rank in range(len(board)):
        for file in range(len(board[0])):
            current_charact = board[rank, file]
            #match current_charact:
            #    case "p":
            #        current_piece = Pawn(False, (rank, file))
            #        board[rank, file] = current_piece
            #        # black_pieces[current_charact] = current_piece
            #    case "r":
            #        current_piece = Rook(False, (rank, file))
            #        board[rank, file] = current_piece
            #        # black_pieces[current_charact] = current_piece
            #    case "n":
            #        current_piece = Knight(False, (rank, file))
            #        board[rank, file] = current_piece
            #        # black_pieces[current_charact] = current_piece
            #    case "b":
            #        current_piece = Bishop(False, (rank, file))
            #        board[rank, file] = current_piece
            #        # black_pieces[current_charact] = current_piece
            #    case "q":
            #        current_piece = Queen(False, (rank, file))
            #        board[rank, file] = current_piece
            #        # black_pieces[current_charact] = current_piece
            #    case "k":
            #        current_piece = King(False, (rank, file))
            #        board[rank, file] = current_piece
            #        # black_pieces[current_charact] = current_piece
            #    case "P":
            #        current_piece = Pawn(True, (rank, file))
            #        board[rank, file] = current_piece
            #        # white_pieces[current_charact] = current_piece
            #    case "R":
            #        current_piece = Rook(True, (rank, file))
            #        board[rank, file] = current_piece
            #        # white_pieces[current_charact] = current_piece
            #    case "N":
            #        current_piece = Knight(True, (rank, file))
            #        board[rank, file] = current_piece
            #        # white_pieces[current_charact] = current_piece
            #    case "B":
            #        current_piece = Bishop(True, (rank, file))
            #        board[rank, file] = current_piece
            #        # white_pieces[current_charact] = current_piece
            #    case "Q":
            #        current_piece = Queen(True, (rank, file))
            #        board[rank, file] = current_piece
            #        # white_pieces[current_charact] = current_piece
            #    case "K":
            #        current_piece = King(True, (rank, file))
            #        board[rank, file] = current_piece
            #        # white_pieces[current_charact] = current_piece
            if current_charact == "p":
                current_piece = Pawn(False, (rank, file))
                board[rank, file] = current_piece
            elif current_charact == "r":
                current_piece = Rook(False, (rank, file))
                board[rank, file] = current_piece
            elif current_charact == "n":
                current_piece = Knight(False, (rank, file))
                board[rank, file] = current_piece
            elif current_charact == "b":
                current_piece = Bishop(False, (rank, file))
                board[rank, file] = current_piece
            elif current_charact == "q":
                current_piece = Queen(False, (rank, file))
                board[rank, file] = current_piece
            elif current_charact == "k":
                current_piece = King(False, (rank, file))
                board[rank, file] = current_piece
            elif current_charact == "P":
                current_piece = Pawn(True, (rank, file))
                board[rank, file] = current_piece
            elif current_charact == "R":
                current_piece = Rook(True, (rank, file))
                board[rank, file] = current_piece
            elif current_charact == "N":
                current_piece = Knight(True, (rank, file))
                board[rank, file] = current_piece
            elif current_charact == "B":
                current_piece = Bishop(True, (rank, file))
                board[rank, file] = current_piece
            elif current_charact == "Q":
                current_piece = Queen(True, (rank, file))
                board[rank, file] = current_piece
            elif current_charact == "K":
                current_piece = King(True, (rank, file))
                board[rank, file] = current_piece

    return board  # , white_pieces, black_pieces


def is_in_check(b: Board, checked_side: bool) -> bool:
    '''return True if the king of %checked_side% is in check; return False otherwise'''

    # for _r in b.board:
    #     for square in _r:
    #         if isinstance(square, Piece) and (square.color is not checked_side):
    #             moves = square.all_moves(b)
    #             for move in moves:
    #                 kingtaken = True
    #                 for square in move.board.flatten():
    #                     if isinstance(square, King) and square.color is checked_side:
    #                         kingtaken = False
    #                         break
    #                 if kingtaken:
    #                     return True
    # return False
    board = b.board
    king_rank = 10
    for x in range(8):
       if king_rank != 10:
           break
       for y in range(8):
           if board[x, y] == King(checked_side, (x, y)):
               king_rank = x
               king_file = y
               break
    if king_rank == 10:
       print(b.to_fen())
       raise Exception("no King found during is_in_check!")
        

    # pawn check
    if checked_side and king_rank >= 2:
        for f in [king_file - 1, king_file + 1]:
            if 0 <= f <= 7 and board[king_rank - 1, f] == Pawn(not checked_side, (king_rank - 1, f)):
                return True
    if not checked_side and king_rank <= 5:
        for f in [king_file - 1, king_file + 1]:
            if 0<= f <= 7 and board[king_rank + 1, f] == Pawn(not checked_side, (king_rank + 1, f)):
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
        if 0 <= p[0] <= 7 and 0 <= p[1] <= 7:
            if board[p] == Knight(not checked_side, p):
                return True

    # diagonal check: bishop or Queen
    is_diagonal = lambda _r, _f: (board[_r, _f] == Bishop(not checked_side, (_r, _f)) or 
                                  board[_r, _f] == Queen(not checked_side, (_r, _f)))
    for i in range(1, 8):
        if king_rank + i > 7 or king_file + i > 7: break
        if is_diagonal(king_rank + i, king_file + i): return True
        if board[king_rank + i, king_file + i] != None: break
    for i in range(1, 8):
        if king_rank + i > 7 or king_file - i < 0: break
        if is_diagonal(king_rank + i, king_file - i): return True
        if board[king_rank + i, king_file - i] != None: break
    for i in range(1, 8):
        if king_rank - i < 0 or king_file - i < 0: break
        if is_diagonal(king_rank - i, king_file - i): return True
        if board[king_rank - i, king_file - i] != None: break
    for i in range(1, 8):
        if king_rank - i < 0 or king_file + i > 7: break
        if is_diagonal(king_rank - i, king_file + i): return True
        if board[king_rank - i, king_file + i] != None: break

    # linear check: rook or Queen
    is_linear = lambda _r, _f: (board[_r, _f] == Rook(not checked_side, (_r, _f)) or 
                                board[_r, _f] == Queen(not checked_side, (_r, _f)))
    for r in range(king_rank + 1, 8):
        if is_linear(r, king_file): return True
        if board[r, king_file] != None: break
    for R in reversed(range(king_rank)):
        if is_linear(R, king_file): return True
        if board[R, king_file] != None: break
    for f in range(king_file + 1, 8):
        if is_linear(king_rank, f): return True
        if board[king_rank, f] != None: break
    for F in reversed(range(king_file)):
        if is_linear(king_rank, F): return True
        if board[king_rank, F] != None: break

    # king check
    for i in [king_rank, king_rank + 1, king_rank - 1]:
        for j in [king_file, king_file + 1, king_file - 1]:
            if 0 <= i <= 7 and 0 <= j <= 7 and board[i, j] == King(not checked_side, (i, j)):
                return True

    return False


def move_generation(b: Board) -> list:
    actions = []  # a list of Board type objects
    num_pieces = 64 - (b.board == None).sum()
    for r in b.board:
        for square in r:
            if issubclass(type(square), Piece) and (square.color is b.turn):
                moves = square.all_moves(b)
                if isinstance(square, Pawn):
                    for move in moves:
                        move.halfmoves = -1
                actions += moves
    r_actions = []
    for action in actions:
        if not is_in_check(action, action.turn):
            if action.turn is False: action.fullmoves += 1  # increment the fullmove clock after black moves
            action.turn = not action.turn
            if action.enpassant == b.enpassant:
                action.enpassant = None
            if 64 - (action.board == None).sum() != num_pieces:
                action.halfmoves = -1
            action.halfmoves += 1
            action.is_expanded = False
            action.is_terminal = None
            action.fen = ''
            action.possible_moves = []
            r_actions.append(copy.deepcopy(action))
    del actions
    return np.random.permutation(r_actions)


def is_mate(b: Board) -> bool:
    b.expand()
    return len(b.possible_moves) == 0


def result(b: Board) -> int:
    '''
    draw: return 0
    white wins: return 1
    black wins: return -1
    '''
    assert(b.check_terminal())
    if is_in_check(b, b.turn):
        return -1 if b.turn else 1
    return 0
