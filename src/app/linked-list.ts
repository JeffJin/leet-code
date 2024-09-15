export class ListNode {
  val: number
  next: ListNode | null;
  prev: ListNode | null;

  constructor(val?: number, next?: ListNode | null, prev?: ListNode | null) {
    this.val = (val === undefined ? 0 : val)
    this.next = (next === undefined ? null : next)
    this.prev = (prev === undefined ? null : prev)
  }
}

class MyLinkedList {
  private list: ListNode[] = [];
  constructor() {

  }

  get(index: number): number {
    return this.list[index].val;
  }

  addAtHead(val: number): void {
    let item = new ListNode(val);
    if(this.list.length == 0) {
      this.list.push(item);
    } else {
      item.next = this.list[0];
      this.list.unshift(item);
    }
  }

  addAtTail(val: number): void {
    let item = new ListNode(val);
    if(this.list.length == 0) {
      this.list.push(item);
    } else {
      this.list[this.list.length - 1].next = item;
      this.list.push(item);
    }
  }

  addAtIndex(index: number, val: number): void {
    let prev = null;
    let item = new ListNode(val);
    if(index == this.list.length){
      this.list[this.list.length - 1].next = item;
      this.list.push(item);
      return;
    }
    let curr = this.list[index];
    if(index > 0) {
      prev = this.list[index - 1];
    }
    if(prev) {
      prev.next = item;
    }
    item.next = curr;
    this.list = [...this.list.slice(0, index), item, ...this.list.slice(index)];
  }

  deleteAtIndex(index: number): void {
    let prev = null;
    let curr = this.list[index];
    let next = curr.next;
    if(index > 0) {
      prev = this.list[index - 1];
    }
    if(prev) {
      prev.next = next;
      curr.next = null;
      this.list = [...this.list.slice(0, index-1), ...this.list.slice(index)];
    } else {
      let item = this.list.shift();
      item!.next = null;
    }
  }
}

export const buildLinkedList = (nums: number[]): ListNode | null => {
  let head = new ListNode(nums[0]);
  let node = head;
  for(let i = 1; i < nums.length; i++) {
    node.next = new ListNode(nums[i]);
    node = node.next;
  }
  return head;
}
