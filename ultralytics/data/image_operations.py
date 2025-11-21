import cv2
import numpy as np
import torch


class Shuffler:
    def __init__(self, tile_shape=None, num_oper=None, operations=None):  # Tile in shape of (B, C, H, W)
        if operations is not None:
            self.operations = operations
        else:
            self.operations = self._create_operations(tile_shape, num_oper)

    def _create_operations(self, tile_shape, num_oper):
        H, W = tile_shape
        operations = []
        for _ in range(num_oper):
            y_ran = np.random.randint(0, H)
            delta_x_top_ran = np.random.randint(0, W)
            delta_x_bottom_ran = np.random.randint(0, W)
            smax = SliceAndMoveX(
                y=y_ran, delta_x_top=delta_x_top_ran, delta_x_bottom=delta_x_bottom_ran
            )
            operations.append(smax)
        for _ in range(num_oper):
            x_ran = np.random.randint(0, W)
            delta_y_left_ran = np.random.randint(0, H)
            delta_y_right_ran = np.random.randint(0, H)
            smay = SliceAndMoveY(
                x=x_ran, delta_y_left=delta_y_left_ran, delta_y_right=delta_y_right_ran
            )
            operations.append(smay)
        np.random.shuffle(operations)
        return operations

    def scale(self, scale: tuple[int, int]):
        scale_h, scale_w = scale
        scaled_operations = []
        for operation in self.operations:
            scaled_operation = operation.scale(scale_h, scale_w)
            scaled_operations.append(scaled_operation)
        return Shuffler(operations=scaled_operations)

    def shuffle(self, tile):
        for operation in self.operations:
            tile = operation.apply(tile)
        return tile

    def unshuffle(self, tile):
        for operation in reversed(self.operations):
            tile = operation.reverse(tile)
        return tile


class SliceAndMove:
    def __init__(
        self, split_pos: int, delta_first: int, delta_second: int, split_dim: int, move_dim: int
    ):
        self.split_pos = split_pos
        self.delta_first = delta_first
        self.delta_second = delta_second
        self.split_dim = split_dim
        self.move_dim = move_dim

    def apply(self, tile_tensor):  # Tile tensor has a shape of B, C, H, W
        first = self._slice_first(tile_tensor)
        second = self._slice_second(tile_tensor)
        first = torch.roll(first, shifts=self.delta_first, dims=self.move_dim)
        second = torch.roll(second, shifts=self.delta_second, dims=self.move_dim)
        return torch.cat([first, second], dim=self.split_dim)

    def reverse(self, tile_tensor, scale=None):
        first = self._slice_first(tile_tensor)
        second = self._slice_second(tile_tensor)
        first = torch.roll(first, shifts=-self.delta_first, dims=self.move_dim)
        second = torch.roll(second, shifts=-self.delta_second, dims=self.move_dim)
        combined = torch.cat([first, second], dim=self.split_dim)
        return combined

    def _slice_first(self, tile_tensor):
        if self.split_dim == 2:
            return tile_tensor[:, :, : self.split_pos, :]
        else:
            return tile_tensor[:, :, :, : self.split_pos]

    def _slice_second(self, tile_tensor):
        if self.split_dim == 2:
            return tile_tensor[:, :, self.split_pos :, :]
        else:
            return tile_tensor[:, :, :, self.split_pos :]


class SliceAndMoveX(SliceAndMove):
    def __init__(self, y: int, delta_x_top: int, delta_x_bottom: int):
        split_dim = 2  # Split along Y-axis
        move_dim = 3  # Move along X-axis
        super().__init__(
            split_pos=y,
            delta_first=delta_x_top,
            delta_second=delta_x_bottom,
            split_dim=split_dim,
            move_dim=move_dim,
        )

    def scale(self, scale_h: int, scale_w: int):
        return SliceAndMoveX(
            y=int(self.split_pos * scale_h),
            delta_x_top=int(self.delta_first * scale_w),
            delta_x_bottom=int(self.delta_second * scale_w),
        )


class SliceAndMoveY(SliceAndMove):
    def __init__(self, x: int, delta_y_left: int, delta_y_right: int):
        split_dim = 3  # Split along X-axis
        move_dim = 2  # Move along Y-axis
        super().__init__(
            split_pos=x,
            delta_first=delta_y_left,
            delta_second=delta_y_right,
            split_dim=split_dim,
            move_dim=move_dim,
        )

    def scale(self, scale_h: int, scale_w: int):
        return SliceAndMoveY(
            x=int(self.split_pos * scale_w),
            delta_y_left=int(self.delta_first * scale_h),
            delta_y_right=int(self.delta_second * scale_h),
        )
