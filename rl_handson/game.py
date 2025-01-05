from typing import List


class ConnectFour:
    def __init__(
        self, rows: int = 6, cols: int = 7, bits_in_len: int = 3, count_to_win: int = 4
    ):
        self.rows = rows
        self.cols = cols
        self.bits_in_len = bits_in_len
        self.count_to_win = count_to_win
        self.player_black = 1
        self.player_white = 0
        self.initial_state = self.encode_lists([[]] * self.cols)
        # 2 channels: 0 for empty, 1 for black, 2 for white
        self.obs_shape = (2, self.rows, self.cols)

    def bits_to_int(self, bits) -> int:
        res = 0
        for b in bits:
            res *= 2
            res += b
        return res

    def int_to_bits(self, num, bits):
        res = []
        for _ in range(bits):
            res.append(num % 2)
            num //= 2
        return res[::-1]

    def encode_lists(self, field_lists: List[List[int]]) -> int:
        """
        Encode lists representation into the binary numbers
        :param field_lists: list of self.cols lists with 0s and 1s
        :return: integer number with encoded game state
        """
        assert isinstance(field_lists, list)
        assert len(field_lists) == self.cols

        bits = []
        len_bits = []
        for col in field_lists:
            bits.extend(col)
            free_len = self.rows - len(col)
            bits.extend([0] * free_len)
            len_bits.extend(self.int_to_bits(free_len, bits=self.bits_in_len))
        bits.extend(len_bits)
        return self.bits_to_int(bits)

    def decode_binary(self, state_int):
        """
        Decode binary representation into the list view
        :param state_int: integer representing the field
        :return: list of self.cols lists
        """
        assert isinstance(state_int, int)
        bits = self.int_to_bits(
            state_int, bits=self.cols * self.rows + self.cols * self.bits_in_len
        )
        res = []
        len_bits = bits[self.cols * self.rows :]
        for col in range(self.cols):
            vals = bits[col * self.rows : (col + 1) * self.rows]
            lens = self.bits_to_int(
                len_bits[col * self.bits_in_len : (col + 1) * self.bits_in_len]
            )
            if lens > 0:
                vals = vals[:-lens]
            res.append(vals)
        return res

    def possible_moves(self, state_int):
        """
        This function could be calculated directly from bits, but I'm too lazy
        :param state_int: field representation
        :return: the list of columns which we can make a move
        """
        assert isinstance(state_int, int)
        field = self.decode_binary(state_int)
        return [idx for idx, col in enumerate(field) if len(col) < self.rows]

    def _check_won(self, field, col, delta_row):
        """
        Check for horizontal/diagonal win condition for the last player moved in the column
        :param field: list of lists
        :param col: column index
        :param delta_row: if 0, checks for horizontal won, 1 for rising diagonal, -1 for falling
        :return: True if won, False if not
        """
        player = field[col][-1]
        coord = len(field[col]) - 1
        total = 1
        # negative dir
        cur_coord = coord - delta_row
        for c in range(col - 1, -1, -1):
            if len(field[c]) <= cur_coord or cur_coord < 0 or cur_coord >= self.rows:
                break
            if field[c][cur_coord] != player:
                break
            total += 1
            if total == self.count_to_win:
                return True
            cur_coord -= delta_row
        # positive dir
        cur_coord = coord + delta_row
        for c in range(col + 1, self.cols):
            if len(field[c]) <= cur_coord or cur_coord < 0 or cur_coord >= self.rows:
                break
            if field[c][cur_coord] != player:
                break
            total += 1
            if total == self.count_to_win:
                return True
            cur_coord += delta_row
        return False

    def move(self, state_int, col, player):
        """
        Perform move into given column. Assume the move could be performed, otherwise, assertion will be raised
        :param state_int: current state
        :param col: column to make a move
        :param player: player index (player_white or player_black)
        :return: tuple of (state_new, won). Value won is bool, True if this move lead
        to victory or False otherwise (but it could be a draw)
        """
        assert isinstance(state_int, int)
        assert isinstance(col, int)
        assert 0 <= col < self.cols
        assert player == self.player_black or player == self.player_white
        field = self.decode_binary(state_int)
        assert len(field[col]) < self.rows
        field[col].append(player)
        # check for victory: the simplest vertical case
        suff = field[col][-self.count_to_win :]
        won = suff == [player] * self.count_to_win
        if not won:
            won = (
                self._check_won(field, col, 0)
                or self._check_won(field, col, 1)
                or self._check_won(field, col, -1)
            )
        state_new = self.encode_lists(field)
        return state_new, won

    def render(self, state_int):
        state_list = self.decode_binary(state_int)
        data = [[" "] * self.cols for _ in range(self.rows)]
        for col_idx, col in enumerate(state_list):
            for rev_row_idx, cell in enumerate(col):
                row_idx = self.rows - rev_row_idx - 1
                data[row_idx][col_idx] = str(cell)
        return ["".join(row) for row in data]

    def update_counts(self, counts_dict, key, counts):
        v = counts_dict.get(key, (0, 0, 0))
        res = (v[0] + counts[0], v[1] + counts[1], v[2] + counts[2])
        counts_dict[key] = res
