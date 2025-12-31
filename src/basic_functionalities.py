#!/usr/bin/env python3
"""
(2,3)-Shamir Secret Sharing MPC Functionalities (Final Version)
"""

import secrets
import hashlib
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from domain_config import DomainConfig, get_config


@dataclass
class Share:
    """Shamir [CN]"""
    value: int          # [CN]
    party_id: int       # [CN] ID (1, 2, [CN] 3)
    degree: int = 1     # [CN]（1 [CN] 2）
    
    def __repr__(self):
        return f"Share(party={self.party_id}, value={self.value}, deg={self.degree})"
    
    def add_mod(self, other, field_size):
        """[CN]（[CN]+[CN] [CN] [CN]+[CN]）"""
        if isinstance(other, Share):
            return Share((self.value + other.value) % field_size, self.party_id, self.degree)
        else:  # [CN]
            return Share((self.value + other) % field_size, self.party_id, self.degree)
    
    def sub_mod(self, other, field_size):
        """[CN]（[CN]-[CN] [CN] [CN]-[CN]）"""
        if isinstance(other, Share):
            return Share((self.value - other.value) % field_size, self.party_id, self.degree)
        else:  # [CN]
            return Share((self.value - other) % field_size, self.party_id, self.degree)
    
    def mul_mod(self, scalar: int, field_size):
        """[CN]"""
        return Share((self.value * scalar) % field_size, self.party_id, self.degree)


@dataclass
class BeaverTriple:
    """Beaver [CN]"""
    a_shares: List[Share]
    b_shares: List[Share]
    c_shares: List[Share]


class MPC23SSS:
    """
    (2,3)-Shamir [CN] MPC [CN]
    [CN]
    """
    
    def __init__(self, config: Optional[DomainConfig] = None):
        """initialize
        
        Args:
            config: [CN]，[CN] None [CN]
        """
        if config is None:
            config = get_config("laion")
        
        self.config = config
        self.field_size = config.prime
        self.n = 3  # [CN]
        self.t = 1  # [CN]
        
        # [CN] Beaver [CN]
        self.triples = []
        self._generate_triples(100)  # [CN] 100 [CN]
        
    def _mod(self, x: int) -> int:
        """[CN]"""
        return ((x % self.field_size) + self.field_size) % self.field_size
    
    def _generate_triples(self, count: int):
        """[CN] Beaver [CN]"""
        for _ in range(count):
            # [CN] a [CN] b
            a = secrets.randbelow(self.field_size)
            b = secrets.randbelow(self.field_size)
            c = self._mod(a * b)
            
            # [CN] a, b, c
            a_shares = self.share_secret(a)
            b_shares = self.share_secret(b)
            c_shares = self.share_secret(c)
            
            triple = BeaverTriple(a_shares, b_shares, c_shares)
            self.triples.append(triple)
    
    def _get_triple(self) -> BeaverTriple:
        """[CN]"""
        if not self.triples:
            self._generate_triples(10)  # [CN]
        return self.triples.pop()
    
    def share_secret(self, secret: int, degree: int = 1) -> List[Share]:
        """
        [CN] (t,n)-Shamir [CN]
        """
        secret = self._mod(secret)
        
        # [CN]
        coeffs = [secret]
        for _ in range(degree):
            coeffs.append(secrets.randbelow(self.field_size))
        
        # calculate[CN]
        shares = []
        for party_id in range(1, self.n + 1):
            value = 0
            x_power = 1
            for coeff in coeffs:
                value = self._mod(value + self._mod(coeff * x_power))
                x_power = self._mod(x_power * party_id)
            shares.append(Share(value, party_id, degree))
        
        return shares
    
    def reconstruct(self, shares: List[Share]) -> int:
        """
        [CN]（Lagrange [CN]）
        """
        if len(shares) < self.t + 1:
            raise ValueError(f"[CN] {self.t + 1} [CN]")
        
        # [CN] t+1 [CN]
        used_shares = shares[:self.t + 1]
        
        result = 0
        for i, share_i in enumerate(used_shares):
            # calculate Lagrange [CN] x=0 [CN]
            numerator = 1
            denominator = 1
            
            for j, share_j in enumerate(used_shares):
                if i != j:
                    numerator = self._mod(numerator * (0 - share_j.party_id))
                    denominator = self._mod(denominator * (share_i.party_id - share_j.party_id))
            
            # calculate[CN]
            inv_denominator = pow(denominator, self.field_size - 2, self.field_size)
            lagrange_coeff = self._mod(numerator * inv_denominator)
            
            # [CN]
            result = self._mod(result + self._mod(share_i.value * lagrange_coeff))
        
        return result
    
    # ==================== MPC [CN] ====================
    
    # def F_Rand(self) -> List[Share]:
    #     """F.Rand() - [CN]"""
    #     random_value = secrets.randbelow(self.field_size)
    #     return self.share_secret(random_value)
    
    def F_DRand(self) -> List[Share]:
        """F.DRand() - [CN] degree-2 [CN]"""
        random_value = secrets.randbelow(self.field_size)
        return self.share_secret(random_value, degree=2)
    
    # def F_Zero(self) -> List[Share]:
    #     """F.Zero() - [CN] 0 [CN]"""
    #     return self.share_secret(0)
    
    def F_Mult_Simple(self, x_shares: List[Share], y_shares: List[Share]) -> List[Share]:
        """
        [CN]
        [CN]（[CN]）
        """
        # [CN] x [CN] y
        x = self.reconstruct(x_shares)
        y = self.reconstruct(y_shares)
        
        # calculate[CN]
        product = self._mod(x * y)
        
        # [CN]
        return self.share_secret(product)
    
    def F_Mult_BGW_Incorrect(self, x_shares: List[Share], y_shares: List[Share]) -> List[Share]:
        """
        [[CN] - [CN]]
        BGW[CN]，[CN]
        [CN]，[CN] F_Mult [CN] F_Mult_Beaver
        """
        # Step 1: [CN]
        # [CN]calculate z_i = x_i * y_i
        # [CN] degree-2 [CN]
        z_shares_deg2 = []
        for i in range(self.n):
            z_value = self._mod(x_shares[i].value * y_shares[i].value)
            z_shares_deg2.append(Share(z_value, i + 1, degree=2))
        
        # Step 2: [CN]（[CN]）
        # [CN]
        
        all_reshares = []
        for i in range(self.n):
            reshared = self.share_secret(z_shares_deg2[i].value, degree=1)
            all_reshares.append(reshared)
        
        final_shares = []
        for party_id in range(self.n):
            sum_value = 0
            for i in range(self.n):
                sum_value = self._mod(sum_value + all_reshares[i][party_id].value)
            final_shares.append(Share(sum_value, party_id + 1, degree=1))
        
        return final_shares
    
    # def F_Mult_Beaver(self, x_shares: List[Share], y_shares: List[Share]) -> List[Share]:
    #     """
    #     [CN] Beaver [CN]（[CN]）
    #     [CN]
    #     """
    #     # [CN]
    #     triple = self._get_triple()
    #     
    #     # Step 1: calculate [e] = [x] - [a] [CN] [f] = [y] - [b]
    #     e_shares = []
    #     f_shares = []
    #     for i in range(self.n):
    #         e_shares.append(x_shares[i].sub_mod(triple.a_shares[i], self.field_size))
    #         f_shares.append(y_shares[i].sub_mod(triple.b_shares[i], self.field_size))
    #     
    #     # Step 2: [CN] e [CN] f
    #     e = self.Open(e_shares)
    #     f = self.Open(f_shares)
    #     
    #     # Step 3: calculate [xy] = [c] + e[b] + f[a] + ef
    #     xy_shares = []
    #     for i in range(self.n):
    #         # [xy]ᵢ = [c]ᵢ + e·[b]ᵢ + f·[a]ᵢ + e·f
    #         value = triple.c_shares[i].value
    #         value = self._mod(value + e * triple.b_shares[i].value)
    #         value = self._mod(value + f * triple.a_shares[i].value)
    #         value = self._mod(value + e * f)
    #         xy_shares.append(Share(value, i + 1))
    #     
    #     return xy_shares
    
    # def F_Mult(self, x_shares: List[Share], y_shares: List[Share]) -> List[Share]:
    #     """
    #     F.Mult([x], [y]) - [CN]
    #     [CN] Beaver [CN]
    #     """
    #     return self.F_Mult_Beaver(x_shares, y_shares)
    
    # def F_SoP(self, x_vector: List[List[Share]], y_vector: List[List[Share]]) -> List[Share]:
    #     """F.SoP([x], [y]) - [CN]"""
    #     if len(x_vector) != len(y_vector):
    #         raise ValueError("[CN]")
    #     
    #     # initialize[CN]
    #     result = self.F_Zero()
    #     
    #     # [CN]
    #     for i in range(len(x_vector)):
    #         # calculate x[i] * y[i]
    #         product = self.F_Mult(x_vector[i], y_vector[i])
    #         
    #         # [CN]：result = result + product
    #         for j in range(self.n):
    #             result[j].value = self._mod(result[j].value + product[j].value)
    #     
    #     return result
    
    # def F_CheckZero(self, x_shares: List[Share]) -> List[Share]:
    #     """F.CheckZero([x]) - [CN]"""
    #     # [CN] r
    #     r_shares = self.F_Rand()
    #     
    #     # calculate z = r * x
    #     z_shares = self.F_Mult(r_shares, x_shares)
    #     
    #     # [CN] z
    #     z_value = self.Open(z_shares)
    #     
    #     # return[CN]
    #     if z_value == 0:
    #         return self.share_secret(1)
    #     else:
    #         return self.share_secret(0)
    
    def Open(self, shares: List[Share]) -> int:
        """
        Open([x]) - [CN]
        [CN]
        """
        if len(shares) != self.n:
            raise ValueError(f"[CN] {self.n} [CN]")
        
        # [CN]：[CN] 2-subset [CN]
        # [CN] (2,3)-SSS，[CN] C(3,2) = 3 [CN]
        
        values = []
        
        # (Party 1, Party 2)
        values.append(self.reconstruct([shares[0], shares[1]]))
        
        # (Party 1, Party 3)
        values.append(self.reconstruct([shares[0], shares[2]]))
        
        # (Party 2, Party 3)
        values.append(self.reconstruct([shares[1], shares[2]]))
        
        # [CN]
        if values[0] == values[1] == values[2]:
            return values[0]
        
        # [CN]，[CN]
        from collections import Counter
        counter = Counter(values)
        most_common = counter.most_common(1)[0]
        
        if most_common[1] >= 2:
            # [CN]
            return most_common[0]
        else:
            raise ValueError("[CN]：[CN]")
    
    # ==================== [CN] ====================
    
    def print_shares(self, shares: List[Share], name: str = ""):
        """print[CN]"""
        if name:
            print(f"{name}:")
        for share in shares:
            print(f"  {share}")


def test_basic_operations():
    """[CN]"""
    print("=" * 60)
    print("[CN] (2,3)-SSS [CN]")
    print("=" * 60)
    
    mpc = MPC23SSS()
    
    # Test 1: [CN]
    print("\n1. [CN]")
    secret = 42
    shares = mpc.share_secret(secret)
    
    # [CN]
    result1 = mpc.reconstruct([shares[0], shares[1]])
    result2 = mpc.reconstruct([shares[0], shares[2]])
    result3 = mpc.reconstruct([shares[1], shares[2]])
    
    print(f"   [CN]: {secret}")
    print(f"   [CN] (1,2): {result1}")
    print(f"   [CN] (1,3): {result2}")
    print(f"   [CN] (2,3): {result3}")
    print(f"   ✓ [CN]" if result1 == result2 == result3 == secret else "   ✗ [CN]")
    
    # Test 2: [CN]
    print("\n2. [CN]")
    x = 15
    y = 27
    x_shares = mpc.share_secret(x)
    y_shares = mpc.share_secret(y)
    
    # [CN]
    sum_shares = []
    for i in range(3):
        sum_value = mpc._mod(x_shares[i].value + y_shares[i].value)
        sum_shares.append(Share(sum_value, i + 1))
    
    result = mpc.reconstruct(sum_shares[:2])
    print(f"   {x} + {y} = {result} ([CN]: {x + y})")
    print(f"   ✓ [CN]" if result == x + y else "   ✗ [CN]")


def test_mpc_functionalities():
    """[CN] MPC [CN]"""
    print("\n\n" + "=" * 60)
    print("[CN] MPC [CN]")
    print("=" * 60)
    
    mpc = MPC23SSS()
    
    # Test 1: F.Rand()
    print("\n1. F.Rand() - [CN]")
    rand_shares = mpc.F_Rand()
    rand_value = mpc.Open(rand_shares)
    print(f"   [CN]: {rand_value}")
    
    # Test 2: F.Zero()
    print("\n2. F.Zero() - [CN]")
    zero_shares = mpc.F_Zero()
    zero_value = mpc.Open(zero_shares)
    print(f"   [CN]: {zero_value}")
    print(f"   ✓ [CN]" if zero_value == 0 else "   ✗ [CN]")
    
    # Test 3: F.Mult() - [CN]
    print("\n3. F.Mult() - [CN]")
    test_cases = [(3, 4), (5, 7), (10, 10)]
    
    for x, y in test_cases:
        x_shares = mpc.share_secret(x)
        y_shares = mpc.share_secret(y)
        
        # [CN] Beaver [CN]
        xy_shares = mpc.F_Mult(x_shares, y_shares)
        xy_value = mpc.Open(xy_shares)
        
        print(f"   {x} × {y} = {xy_value} ([CN]: {x*y})")
        print(f"   ✓ [CN]" if xy_value == x*y else "   ✗ [CN]")
    
    # Test 4: F.SoP() - [CN]
    print("\n4. F.SoP() - [CN]")
    # [CN]
    x_vec = [mpc.share_secret(2), mpc.share_secret(3)]
    y_vec = [mpc.share_secret(4), mpc.share_secret(5)]
    
    dot_shares = mpc.F_SoP(x_vec, y_vec)
    dot_value = mpc.Open(dot_shares)
    expected = 2*4 + 3*5  # = 23
    
    print(f"   [2,3] · [4,5] = {dot_value} ([CN]: {expected})")
    print(f"   ✓ [CN]" if dot_value == expected else "   ✗ [CN]")
    
    # Test 5: F.CheckZero()
    print("\n5. F.CheckZero() - [CN]")
    
    # [CN]
    zero_shares = mpc.share_secret(0)
    is_zero_shares = mpc.F_CheckZero(zero_shares)
    is_zero = mpc.Open(is_zero_shares)
    print(f"   CheckZero(0) = {is_zero}")
    print(f"   ✓ [CN]" if is_zero == 1 else "   ✗ [CN]")
    
    # [CN]
    nonzero_shares = mpc.share_secret(5)
    is_zero_shares = mpc.F_CheckZero(nonzero_shares)
    is_zero = mpc.Open(is_zero_shares)
    print(f"   CheckZero(5) = {is_zero}")
    print(f"   ✓ [CN]" if is_zero == 0 else "   ✗ [CN]")


def test_beaver_triple_generation():
    """[CN] Beaver [CN]"""
    print("\n\n" + "=" * 60)
    print("[CN] Beaver [CN]")
    print("=" * 60)
    
    mpc = MPC23SSS()
    
    print("\n1. [CN]")
    # [CN]
    triple = mpc._get_triple()
    
    # [CN] a, b, c
    a = mpc.Open(triple.a_shares)
    b = mpc.Open(triple.b_shares)
    c = mpc.Open(triple.c_shares)
    
    print(f"   [CN]: a={a}, b={b}, c={c}")
    print(f"   [CN] c = a × b: {c} = {a} × {b}")
    print(f"   ✓ [CN]" if c == mpc._mod(a * b) else "   ✗ [CN]")
    
    print("\n2. [CN] Beaver [CN]")
    x = 15
    y = 20
    expected = x * y
    
    x_shares = mpc.share_secret(x)
    y_shares = mpc.share_secret(y)
    
    # [CN]
    triple2 = mpc._get_triple()
    
    # [CN]calculate e = x - a, f = y - b
    e_shares = []
    f_shares = []
    for i in range(3):
        e_shares.append(x_shares[i].sub_mod(triple2.a_shares[i], mpc.field_size))
        f_shares.append(y_shares[i].sub_mod(triple2.b_shares[i], mpc.field_size))
    
    e = mpc.Open(e_shares)
    f = mpc.Open(f_shares)
    
    print(f"\n   calculate {x} × {y}:")
    print(f"   e = x - a = {e}")
    print(f"   f = y - b = {f}")
    
    # [CN]
    result_shares = mpc.F_Mult(x_shares, y_shares)
    result = mpc.Open(result_shares)
    print(f"   [CN]: {result} ([CN]: {expected})")
    print(f"   ✓ [CN]" if result == expected else "   ✗ [CN]")


def test_malicious_detection():
    """[CN]"""
    print("\n\n" + "=" * 60)
    print("[CN]")
    print("=" * 60)
    
    mpc = MPC23SSS()
    
    # [CN]
    print("\n1. [CN]")
    x_shares = mpc.share_secret(100)
    x_value = mpc.Open(x_shares)
    print(f"   [CN]: {x_value}")
    print(f"   ✓ [CN]")
    
    # [CN]
    print("\n2. [CN]")
    y_shares = mpc.share_secret(200)
    
    # Party 1 [CN]
    original = y_shares[0].value
    y_shares[0].value = (y_shares[0].value + 999999) % mpc.field_size
    
    try:
        y_value = mpc.Open(y_shares)
        print(f"   [CN]: {y_value}")
        
        # [CN]
        y_shares[0].value = original
        correct_value = mpc.Open(y_shares)
        
        if y_value != correct_value:
            print(f"   ✓ [CN]：[CN] {y_value} != [CN] {correct_value}")
        else:
            print(f"   ✗ [CN]")
    except ValueError as e:
        print(f"   ✓ [CN]: {e}")


def demonstrate_real_application():
    """[CN]"""
    print("\n\n" + "=" * 60)
    print("[CN]：[CN]calculate")
    print("=" * 60)
    
    mpc = MPC23SSS()
    
    print("\n[CN]：[CN]calculate[CN]，[CN]")
    
    # [CN]（[CN]）
    dept1_salary = 500000  # [CN]1
    dept2_salary = 750000  # [CN]2
    dept3_salary = 600000  # [CN]3
    
    # [CN]（[CN]）
    dept1_count = 10
    dept2_count = 15
    dept3_count = 12
    
    print(f"\n[CN]（[CN]）：")
    print(f"   [CN]1: {dept1_count} [CN]")
    print(f"   [CN]2: {dept2_count} [CN]")
    print(f"   [CN]3: {dept3_count} [CN]")
    
    # [CN]
    salary1_shares = mpc.share_secret(dept1_salary)
    salary2_shares = mpc.share_secret(dept2_salary)
    salary3_shares = mpc.share_secret(dept3_salary)
    
    # calculate[CN]（[CN]）
    total_salary_shares = []
    for i in range(3):
        total = salary1_shares[i].value + salary2_shares[i].value + salary3_shares[i].value
        total_salary_shares.append(Share(mpc._mod(total), i + 1))
    
    # [CN]
    total_salary = mpc.Open(total_salary_shares)
    total_count = dept1_count + dept2_count + dept3_count
    
    # calculate[CN]
    average_salary = total_salary // total_count
    
    print(f"\ncalculate[CN]：")
    print(f"   [CN]: {total_salary}")
    print(f"   [CN]: {total_count}")
    print(f"   [CN]: {average_salary}")
    
    # [CN]
    actual_total = dept1_salary + dept2_salary + dept3_salary
    actual_average = actual_total // total_count
    
    print(f"\n[CN]：")
    print(f"   ✓ [CN]" if average_salary == actual_average else "   ✗ [CN]")
    print(f"\n[CN]：[CN]！")


if __name__ == "__main__":
    test_basic_operations()
    test_mpc_functionalities()
    test_beaver_triple_generation()
    test_malicious_detection()
    demonstrate_real_application()
    
    print("\n\n" + "=" * 60)
    print("💡 [CN]")
    print("=" * 60)
    print("\n✅ [CN]：")
    print("  - (2,3)-Shamir [CN]")
    print("  - Beaver [CN]（[CN]）")
    print("  - [CN] MPC [CN]：[CN]、[CN]、[CN]")
    print("  - [CN]、[CN]")
    print("  - [CN]")
    print("  - [CN]")
    
    print("\n🔧 [CN]：")
    print("  - [CN] Beaver [CN]")
    print("  - [CN]")
    print("  - [CN] Beaver [CN]")
    
    print("\n🚀 [CN]：")
    print("  - [CN]")
    print("  - [CN]calculate")
    print("  - CBDC [CN]")
    print("  - [CN]")