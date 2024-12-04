export class QuickFind {
  private root: number[];
  private rank: number[];

  constructor(size: number) {
    this.root = new Array(size);
    this.rank = new Array(size);
    for (let i = 0; i < size; i++) {
      this.root[i] = i;
      this.rank[i] = 1;
    }
  }

  find(x: number): number {
    if(this.root[x] == x) {
      return x;
    }
    return this.root[x] = this.find(this.root[x]);
  }

  union(x: number, y: number) {
    const rootX = this.find(x);
    const rootY = this.find(y);
    if(rootX != rootY) {
      if(this.rank[rootX] > this.rank[rootY]) {
        this.root[rootY] = rootX;
      } else if(this.rank[rootX] < this.rank[rootY]) {
        this.root[rootX] = rootY;
      } else {
        this.root[rootY] = rootX;
        this.rank[rootX] += 1;
      }
    }
  }

  connected(x: number, y: number): boolean {
    return this.find(x) === this.find(y);
  }
}
