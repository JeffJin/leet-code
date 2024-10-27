export class ArrayReader {
  // Compares the sum of arr[l..r] with the sum of arr[x..y]
  // return 1 if sum(arr[l..r]) > sum(arr[x..y])
  // return 0 if sum(arr[l..r]) == sum(arr[x..y])
  // return -1 if sum(arr[l..r]) < sum(arr[x..y])

  private array: number[];
  private prefix: number[];
  constructor(arr: number[]) {
    this.array = arr;
    this.prefix = [arr[0]];
    for(let i = 1; i < arr.length; i++) {
      this.prefix[i] = this.prefix[i-1] + arr[i];
    }

  }
  compareSub(l: number, r: number, x: number, y: number): number {
    let lSum = this.prefix[r] - this.prefix[l] + this.array[l];
    let xSum = this.prefix[y] - this.prefix[x] + this.array[x];
    if(lSum > xSum) {
      return 1;
    } else if(xSum > lSum) {
      return -1;
    }
    return 0;
  };

  // Returns the length of the array
  length(): number {
    return this.array.length;
  };
}
