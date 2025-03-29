import chess
import logging
from typing import Optional, Tuple, List
from datetime import datetime

class ChessEngine:
    """
    Lớp ChessEngine mô phỏng một engine cờ vua cơ bản, cung cấp các chức năng:
    - Lấy danh sách nước đi hợp lệ.
    - Thực hiện nước đi.
    - Đánh giá bàn cờ.
    """
    
    def __init__(self):
        """
        Initialize a new chess engine with a standard starting position.
        """
        self.board = chess.Board()
        self.move_history = []
        self.setup_logging()
        
    def setup_logging(self):
        """
        Set up logging for the chess engine.
        """
        self.logger = logging.getLogger('ChessEngine')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(f'data/logs/chess_engine_{timestamp}.log')
        console_handler = logging.StreamHandler()
        
        # Create formatters and add it to handlers
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        
        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("Chess Engine initialized with standard starting position")
        
    def make_move(self, move_uci: str) -> Tuple[bool, str]:
        """
        Make a move on the board using UCI notation.
        
        Args:
            move_uci (str): Move in UCI notation (e.g., "e2e4")
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            # Convert UCI move to chess.Move object
            move = chess.Move.from_uci(move_uci)
            
            # Check if move is legal
            if move not in self.board.legal_moves:
                self.logger.warning(f"Illegal move attempted: {move_uci}")
                return False, f"Illegal move: {move_uci}"
            
            # Execute the move
            self.board.push(move)
            self.move_history.append(move)
            
            # Log the move
            self.logger.info(f"Move made: {move_uci}")
            
            # Check for special moves
            if self.board.is_castling(move):
                self.logger.info("Castling move executed")
            elif self.board.is_en_passant(move):
                self.logger.info("En passant capture executed")
            elif self.board.is_promotion(move):
                self.logger.info(f"Pawn promotion to {self.board.piece_at(move.to_square).symbol()}")
            
            return True, f"Move {move_uci} executed successfully"
            
        except ValueError as e:
            self.logger.error(f"Invalid move format: {move_uci}")
            return False, f"Invalid move format: {move_uci}"
        except Exception as e:
            self.logger.error(f"Error making move: {str(e)}")
            return False, f"Error making move: {str(e)}"
    
    def get_legal_moves(self) -> List[str]:
        """
        Get all legal moves in UCI notation.
        
        Returns:
            List[str]: List of legal moves in UCI notation
        """
        return [move.uci() for move in self.board.legal_moves]
    
    def is_game_over(self) -> bool:
        """
        Check if the game is over.
        
        Returns:
            bool: True if game is over, False otherwise
        """
        return self.board.is_game_over()
    
    def get_game_result(self) -> Optional[str]:
        """
        Get the game result if the game is over.
        
        Returns:
            Optional[str]: Game result or None if game is not over
        """
        if not self.is_game_over():
            return None
            
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn else "White"
            return f"Checkmate! {winner} wins!"
        elif self.board.is_stalemate():
            return "Stalemate! Game is a draw."
        elif self.board.is_insufficient_material():
            return "Draw due to insufficient material."
        elif self.board.is_fifty_moves():
            return "Draw due to fifty-move rule."
        elif self.board.is_repetition():
            return "Draw due to repetition."
        else:
            return "Game is over (unknown reason)."
    
    def get_board_state(self) -> str:
        """
        Get a human-readable representation of the board state.
        
        Returns:
            str: Board state as a string
        """
        return str(self.board)
    
    def get_fen(self) -> str:
        """
        Get the current position in FEN notation.
        
        Returns:
            str: Position in FEN notation
        """
        return self.board.fen()
    
    def reset(self):
        """
        Reset the board to the starting position.
        """
        self.board.reset()
        self.move_history.clear()
        self.logger.info("Board reset to starting position")
    
    def get_piece_at(self, square: str) -> Optional[str]:
        """
        Get the piece at a given square.
        
        Args:
            square (str): Square in algebraic notation (e.g., "e4")
            
        Returns:
            Optional[str]: Piece symbol or None if square is empty
        """
        try:
            square_idx = chess.parse_square(square)
            piece = self.board.piece_at(square_idx)
            return piece.symbol() if piece else None
        except ValueError:
            return None
    
    def is_in_check(self) -> bool:
        """
        Check if the current side is in check.
        
        Returns:
            bool: True if current side is in check
        """
        return self.board.is_check()
    
    def get_current_turn(self) -> str:
        """
        Get the current side to move.
        
        Returns:
            str: "White" or "Black"
        """
        return "White" if self.board.turn else "Black"
    
    def get_move_history(self) -> List[str]:
        """
        Get the move history in UCI notation.
        
        Returns:
            List[str]: List of moves in UCI notation
        """
        return [move.uci() for move in self.move_history]
    
    def get_captured_pieces(self) -> List[str]:
        """
        Get a list of captured pieces.
        
        Returns:
            List[str]: List of captured piece symbols
        """
        captured = []
        for piece_type in chess.PIECE_TYPES:
            for color in [True, False]:  # True for white, False for black
                if self.board.pieces(piece_type, color) == 0:
                    piece = chess.Piece(piece_type, color)
                    captured.append(piece.symbol())
        return captured

# Example usage
if __name__ == "__main__":
    # Create a new chess engine
    engine = ChessEngine()
    
    # Print initial board state
    print("Initial board state:")
    print(engine.get_board_state())
    
    # Make some moves
    moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
    for move in moves:
        success, message = engine.make_move(move)
        print(f"\n{message}")
        print("\nBoard state after move:")
        print(engine.get_board_state())
    
    # Get game status
    print("\nGame status:")
    print(f"Is game over? {engine.is_game_over()}")
    print(f"Current turn: {engine.get_current_turn()}")
    print(f"Move history: {engine.get_move_history()}")
    print(f"Captured pieces: {engine.get_captured_pieces()}")
    
    # Get legal moves
    print("\nLegal moves:")
    print(engine.get_legal_moves())
