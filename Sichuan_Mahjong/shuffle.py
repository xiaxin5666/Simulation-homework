import random
import itertools

import tile
n = 108
tiles = []
for i in range(4):
    tiles.append(tile.Tile.tile_list)
tiles = list(itertools.chain(*tiles))
#print(tiles)

random.shuffle(tiles)
#print(tiles)

player_tiles = [[0 for _ in range(13)] for _ in range(4)]
for i in range(4):
    for j in range(13):
        player_tiles[i][j] = tiles.pop(0)

player_tiles[0].append(tiles.pop(0))
#player_tiles[0].sort()

print(player_tiles)