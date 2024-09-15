import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LeetCodeComponent } from './leet-code.component';

describe('LeetCodeComponent', () => {
  let component: LeetCodeComponent;
  let fixture: ComponentFixture<LeetCodeComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ LeetCodeComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(LeetCodeComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
