from typing import List


BUF_SZ = 4


class Solution:
    def __init__(self, data):
        self.idx = 0
        self.data = data
        self.buf4 = [''] * 4
        self.curr_used = 0
        self.curr_read = 0
        self.EOF = False


    def read4(self, buf: List[str]) -> int:
        """reads 4 consecutive characters from the file,
        then writes those characters into the buffer array buf4

        Args:
            buf (List[str]): Destination buffer

        Returns:
            int: The number of actual characters read
        """
        bytes = min(len(self.data) - self.idx, BUF_SZ)
        i = self.idx
        j = 0
        while j < bytes:
            buf[j] = self.data[i]
            i += 1
            j += 1
        self.idx += bytes
        return bytes


    def read(self, buf: List[str], n: int) -> int:
        """reads n characters from the file and store it in the buffer array buf

        Args:
            buf (List[str]): Destination buffer
            n (int): Number of characters to read (int)

        Returns:
            int: The number of actual characters read
        """
        buf4 = [''] * 4
        num_read = 0
        EOF = False
        while num_read < n and not EOF:
            curr_read = self.read4(buf4)
            delta = min(curr_read, n-num_read)
            buf[num_read:num_read+delta] = buf4[:delta]
            num_read += delta
            if curr_read < 4: EOF = True
        return ''.join(buf), num_read


    def readN(self, buf: List[str], n: int) -> int:
        """reads n characters from the file and store it in the buffer array buf

        Args:
            buf (List[str]): Destination buffer
            n (int): Number of characters to read (int)

        Returns:
            int: The number of actual characters read
        """
        num_read = 0
        while num_read < n and not self.EOF:
            if self.curr_used == self.curr_read:
                self.curr_read = self.read4(self.buf4)
                self.curr_used = 0
                if self.curr_read == 0: self.EOF = True
            else:
                delta = min(self.curr_read-self.curr_used, n-num_read)
                buf[num_read:num_read+delta] = self.buf4[self.curr_used:self.curr_used+delta]
                num_read += delta
                self.curr_used += delta
        return ''.join(buf), num_read


if __name__ == "__main__":
    sol = Solution('abc')
    print(sol.read([], 4))

    sol = Solution('abc')
    ans = []
    for i in [1, 2, 1]:
        ans.append(sol.readN([], i))
    print(ans)
