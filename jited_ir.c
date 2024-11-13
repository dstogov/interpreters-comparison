#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <math.h>

#include "common.h"

#include <stddef.h>
#include "ir.h"
#include "ir_builder.h"

#define ir_CONST_STR(str) ir_const_str(ctx, ir_str(ctx, str))

static inline decode_t decode_at_address(const Instr_t* prog, uint32_t addr) {
    assert(addr < PROGRAM_SIZE);
    decode_t result = {0};
    Instr_t raw_instr = prog[addr];
    result.opcode = raw_instr;
    switch (raw_instr) {
    case Instr_Nop:
    case Instr_Halt:
    case Instr_Print:
    case Instr_Swap:
    case Instr_Dup:
    case Instr_Inc:
    case Instr_Add:
    case Instr_Sub:
    case Instr_Mul:
    case Instr_Rand:
    case Instr_Dec:
    case Instr_Drop:
    case Instr_Over:
    case Instr_Mod:
    case Instr_And:
    case Instr_Or:
    case Instr_Xor:
    case Instr_SHL:
    case Instr_SHR:
    case Instr_Rot:
    case Instr_SQRT:
    case Instr_Pick:
        result.length = 1;
        break;
    case Instr_Push:
    case Instr_JNE:
    case Instr_JE:
    case Instr_Jump:
        result.length = 2;
        if (!(addr+1 < PROGRAM_SIZE)) {
            result.length = 1;
            result.opcode = Instr_Break;
            break;
        }
        result.immediate = (int32_t)prog[addr+1];
        break;
    case Instr_Break:
    default: /* Undefined instructions equal to Break */
        result.length = 1;
        result.opcode = Instr_Break;
        break;
    }
    return result;
}

/*** Service routines ***/
static void jit_push(ir_ctx *ctx, ir_ref cpu, ir_ref *stack_overflow, ir_ref v) {
    // JIT: if (pcpu->sp >= STACK_CAPACITY-1) {
    ir_ref sp_addr = ir_ADD_OFFSET(cpu, offsetof(cpu_t, sp));
    ir_ref sp = ir_LOAD_I32(sp_addr);
    ir_ref if_overflow = ir_IF(ir_GE(sp, ir_CONST_I32(STACK_CAPACITY-1)));

    ir_IF_TRUE_cold(if_overflow);
    ir_END_list(*stack_overflow);

    ir_IF_FALSE(if_overflow);

    // JIT: pcpu->stack[++pcpu->sp] = v;
    sp = ir_ADD_I32(sp, ir_CONST_I32(1));
    ir_STORE(sp_addr, sp);
    ir_STORE(ir_ADD_I32(ir_ADD_OFFSET(cpu, offsetof(cpu_t, stack)),
            ir_MUL_I32(sp, ir_CONST_I32(sizeof(uint32_t)))), v);
}

static ir_ref jit_pop(ir_ctx *ctx, ir_ref cpu, ir_ref *stack_underflow) {
    // JIT: if (pcpu->sp < 0) {
    ir_ref sp_addr = ir_ADD_OFFSET(cpu, offsetof(cpu_t, sp));
    ir_ref sp = ir_LOAD_I32(sp_addr);
    ir_ref if_underflow = ir_IF(ir_LT(sp, ir_CONST_I32(0)));

    ir_IF_TRUE_cold(if_underflow);
    ir_END_list(*stack_underflow);

    ir_IF_FALSE(if_underflow);

    //JIT: pcpu->stack[pcpu->sp--];
    ir_ref ret = ir_LOAD_I32(ir_ADD_I32(ir_ADD_OFFSET(cpu, offsetof(cpu_t, stack)),
            ir_MUL_I32(sp, ir_CONST_I32(sizeof(uint32_t)))));
    sp = ir_SUB_I32(sp, ir_CONST_I32(1));
    ir_STORE(sp_addr, sp);

    return ret;
}

static ir_ref jit_pick(ir_ctx *ctx, ir_ref cpu, ir_ref *stack_bound, ir_ref pos) {
    // JIT: if (pcpu->sp - 1 < pos) {
    ir_ref sp = ir_LOAD_I32(ir_ADD_OFFSET(cpu, offsetof(cpu_t, sp)));
    ir_ref if_out = ir_IF(ir_LT(ir_SUB_U32(sp, ir_CONST_U32(1)), pos));

    ir_IF_TRUE_cold(if_out);
    ir_END_list(*stack_bound);

    ir_IF_FALSE(if_out);
    // JIT: pcpu->stack[pcpu->sp - pos];
    return ir_LOAD_I32(ir_ADD_I32(ir_ADD_OFFSET(cpu, offsetof(cpu_t, stack)),
            ir_MUL_I32(ir_SUB_I32(sp, pos), ir_CONST_I32(sizeof(uint32_t)))));
}

typedef struct _jit_label {
    ir_ref inputs; /* number of input edges */
    ir_ref merge;  /* reference of MERGE or "list" of forward inputs */
} jit_label;

static void jit_goto_backward(ir_ctx *ctx, jit_label *label) {
    ir_set_op(ctx, label->merge, ++label->inputs, ir_END());
}

static void jit_goto_forward(ir_ctx *ctx, jit_label *label) {
    ir_END_list(label->merge);
}

static void jit_program(ir_ctx *ctx, const Instr_t *prog, int len) {
    assert(prog);
    ir_ref stack_overflow = IR_UNUSED;
    ir_ref stack_underflow = IR_UNUSED;
    ir_ref stack_bound = IR_UNUSED;
    jit_label *labels = calloc(len, sizeof(jit_label));
    decode_t decoded;

    /* mark goto targets */
    for (int i=0; i < len;) {
        decoded = decode_at_address(prog, i);
        i += decoded.length;
        switch(decoded.opcode) {
        case Instr_JE:
        case Instr_JNE:
        case Instr_Jump:
            labels[i + decoded.immediate].inputs++;
            break;
        }
    }

    ir_START();
    ir_ref tmp1, tmp2, tmp3;
    ir_ref cpu = ir_PARAM(IR_ADDR, "cpu", 1);
    ir_ref printf_func =
        ir_const_func(ctx, ir_str(ctx, "printf"), ir_proto_1(ctx, IR_I32, IR_VARARG_FUNC, IR_ADDR));
    ir_ref rand_func =
        ir_const_func(ctx, ir_str(ctx, "rand"), ir_proto_0(ctx, IR_I32, 0));
    ir_ref sqrt_func =
        ir_const_func(ctx, ir_str(ctx, "sqrt"), ir_proto_1(ctx, IR_DOUBLE, IR_BUILTIN_FUNC, IR_DOUBLE));

    decoded.opcode = Instr_Nop;

    for (int i=0; i < len;) {
        if (labels[i].inputs > 0) {
            if (decoded.opcode != Instr_Jump) {
                labels[i].inputs++;
                jit_goto_forward(ctx, &labels[i]);
            }
            assert(!ctx->control);
            if (labels[i].inputs == 1) {
                tmp1 = ir_emit1(ctx, IR_BEGIN, IR_UNUSED);
            } else {
                tmp1 = ir_emit_N(ctx, IR_MERGE, labels[i].inputs);
            }
            tmp2 = 0;
            tmp3 = labels[i].merge;
            labels[i].merge = ctx->control = tmp1;

            while (tmp3) {
                /* Store forward GOTOs into MERGE */
                tmp2++;
                assert(tmp2 <= labels[i].inputs);
                ir_set_op(ctx, tmp1, tmp2, tmp3);
                ir_insn *insn = &ctx->ir_base[tmp3];
                assert(insn->op == IR_END);
                tmp3 = insn->op2;
                insn->op2 = IR_UNUSED;
            }
            labels[i].inputs = tmp2;
        }

        decoded = decode_at_address(prog, i);
        i += decoded.length;

        switch(decoded.opcode) {
        case Instr_Nop:
            /* Do nothing */
            break;
        case Instr_Halt:
            // JIT: cpu.state = Cpu_Halted;
            ir_STORE(ir_ADD_OFFSET(cpu, offsetof(cpu_t, state)), ir_CONST_I32(Cpu_Halted));
            ir_RETURN(IR_VOID);
            break;
        case Instr_Push:
            jit_push(ctx, cpu, &stack_overflow, ir_CONST_U32(decoded.immediate));
            break;
        case Instr_Print:
            tmp1 = jit_pop(ctx, cpu, &stack_underflow);
            ir_CALL_2(IR_VOID, printf_func, ir_CONST_STR("[%d]\n"), tmp1);
            break;
        case Instr_Swap:
            tmp1 = jit_pop(ctx, cpu, &stack_underflow);
            tmp2 = jit_pop(ctx, cpu, &stack_underflow);
            jit_push(ctx, cpu, &stack_overflow, tmp1);
            jit_push(ctx, cpu, &stack_overflow, tmp2);
            break;
        case Instr_Dup:
            tmp1 = jit_pop(ctx, cpu, &stack_underflow);
            jit_push(ctx, cpu, &stack_overflow, tmp1);
            jit_push(ctx, cpu, &stack_overflow, tmp1);
            break;
        case Instr_Over:
            tmp1 = jit_pop(ctx, cpu, &stack_underflow);
            tmp2 = jit_pop(ctx, cpu, &stack_underflow);
            jit_push(ctx, cpu, &stack_overflow, tmp2);
            jit_push(ctx, cpu, &stack_overflow, tmp1);
            jit_push(ctx, cpu, &stack_overflow, tmp2);
            break;
        case Instr_Inc:
            tmp1 = jit_pop(ctx, cpu, &stack_underflow);
            jit_push(ctx, cpu, &stack_overflow, ir_ADD_U32(tmp1, ir_CONST_U32(1)));
            break;
        case Instr_Add:
            tmp1 = jit_pop(ctx, cpu, &stack_underflow);
            tmp2 = jit_pop(ctx, cpu, &stack_underflow);
            jit_push(ctx, cpu, &stack_overflow, ir_ADD_U32(tmp1, tmp2));
            break;
        case Instr_Sub:
            tmp1 = jit_pop(ctx, cpu, &stack_underflow);
            tmp2 = jit_pop(ctx, cpu, &stack_underflow);
            jit_push(ctx, cpu, &stack_overflow, ir_SUB_U32(tmp1, tmp2));
            break;
        case Instr_Mod:
            tmp1 = jit_pop(ctx, cpu, &stack_underflow);
            tmp2 = jit_pop(ctx, cpu, &stack_underflow);
            // JIT if (tmp2 == 0)
            tmp3 = ir_IF(ir_EQ(tmp2, ir_CONST_U32(0)));

            ir_IF_TRUE_cold(tmp3);
            // JIT: printf("Division by zero\n");
            ir_CALL_1(IR_VOID, printf_func, ir_CONST_STR("Division by zero\n"));
            // JIT: pcpu->state = Cpu_Break;
            ir_STORE(ir_ADD_OFFSET(cpu, offsetof(cpu_t, state)), ir_CONST_I32(Cpu_Break));
            ir_RETURN(IR_VOID);

            ir_IF_FALSE(tmp3);
            jit_push(ctx, cpu, &stack_overflow, ir_MOD_U32(tmp1, tmp2));
            break;

        case Instr_Mul:
            tmp1 = jit_pop(ctx, cpu, &stack_underflow);
            tmp2 = jit_pop(ctx, cpu, &stack_underflow);
            jit_push(ctx, cpu, &stack_overflow, ir_MUL_U32(tmp1, tmp2));
            break;
        case Instr_Rand:
            tmp1 = ir_CALL(IR_I32, rand_func);
            jit_push(ctx, cpu, &stack_overflow, tmp1);
            break;
        case Instr_Dec:
            tmp1 = jit_pop(ctx, cpu, &stack_underflow);
            jit_push(ctx, cpu, &stack_overflow, ir_SUB_U32(tmp1, ir_CONST_U32(1)));
            break;
        case Instr_Drop:
            (void)jit_pop(ctx, cpu, &stack_underflow);
            break;
        case Instr_JE:
            tmp1 = jit_pop(ctx, cpu, &stack_underflow);
            // JIT: if (tmp1 == 0)
            tmp3 = ir_IF(ir_EQ(tmp1, ir_CONST_U32(0)));
            ir_IF_TRUE(tmp3);
            if (decoded.immediate < -decoded.length) {
                jit_goto_backward(ctx, &labels[i + decoded.immediate]);
            } else {
                jit_goto_forward(ctx, &labels[i + decoded.immediate]);
            }
            ir_IF_FALSE(tmp3);
            break;
        case Instr_JNE:
            tmp1 = jit_pop(ctx, cpu, &stack_underflow);
            // JIT: if (tmp1 == 0)
            tmp3 = ir_IF(ir_NE(tmp1, ir_CONST_U32(0)));
            ir_IF_TRUE(tmp3);
            if (decoded.immediate < -decoded.length) {
                jit_goto_backward(ctx, &labels[i + decoded.immediate]);
            } else {
                jit_goto_forward(ctx, &labels[i + decoded.immediate]);
            }
            ir_IF_FALSE(tmp3);
            break;
        case Instr_Jump:
            if (decoded.immediate < -decoded.length) {
                jit_goto_backward(ctx, &labels[i + decoded.immediate]);
            } else {
                jit_goto_forward(ctx, &labels[i + decoded.immediate]);
            }
            break;
        case Instr_And:
            tmp1 = jit_pop(ctx, cpu, &stack_underflow);
            tmp2 = jit_pop(ctx, cpu, &stack_underflow);
            jit_push(ctx, cpu, &stack_overflow, ir_AND_U32(tmp1, tmp2));
            break;
        case Instr_Or:
            tmp1 = jit_pop(ctx, cpu, &stack_underflow);
            tmp2 = jit_pop(ctx, cpu, &stack_underflow);
            jit_push(ctx, cpu, &stack_overflow, ir_OR_U32(tmp1, tmp2));
            break;
        case Instr_Xor:
            tmp1 = jit_pop(ctx, cpu, &stack_underflow);
            tmp2 = jit_pop(ctx, cpu, &stack_underflow);
            jit_push(ctx, cpu, &stack_overflow, ir_XOR_U32(tmp1, tmp2));
            break;
        case Instr_SHL:
            tmp1 = jit_pop(ctx, cpu, &stack_underflow);
            tmp2 = jit_pop(ctx, cpu, &stack_underflow);
            jit_push(ctx, cpu, &stack_overflow, ir_SHL_U32(tmp1, tmp2));
            break;
        case Instr_SHR:
            tmp1 = jit_pop(ctx, cpu, &stack_underflow);
            tmp2 = jit_pop(ctx, cpu, &stack_underflow);
            jit_push(ctx, cpu, &stack_overflow, ir_SHR_U32(tmp1, tmp2));
            break;
        case Instr_Rot:
            tmp1 = jit_pop(ctx, cpu, &stack_underflow);
            tmp2 = jit_pop(ctx, cpu, &stack_underflow);
            tmp3 = jit_pop(ctx, cpu, &stack_underflow);
            jit_push(ctx, cpu, &stack_overflow, tmp1);
            jit_push(ctx, cpu, &stack_overflow, tmp3);
            jit_push(ctx, cpu, &stack_overflow, tmp2);
            break;
        case Instr_SQRT:
            tmp1 = jit_pop(ctx, cpu, &stack_underflow);
            tmp1 = ir_FP2U32(ir_CALL_1(IR_DOUBLE, sqrt_func, ir_INT2D(tmp1)));
            jit_push(ctx, cpu, &stack_overflow, tmp1);
            break;
        case Instr_Pick:
            tmp1 = jit_pop(ctx, cpu, &stack_underflow);
            tmp1 = jit_pick(ctx, cpu, &stack_bound, tmp1);
            jit_push(ctx, cpu, &stack_overflow, tmp1);
            break;
        case Instr_Break:
            if (ctx->control) {
                // JIT: pcpu->state = Cpu_Break;
                ir_STORE(ir_ADD_OFFSET(cpu, offsetof(cpu_t, state)), ir_CONST_I32(Cpu_Break));
                ir_RETURN(IR_VOID);
            }
            i = len;
            break;
        default:
            assert(0 && "Unsupported instruction");
            break;
        }
    }

    if (stack_overflow) {
        ir_MERGE_list(stack_overflow);
        // JIT: printf("Stack overflow\n");
        ir_CALL_1(IR_VOID, printf_func, ir_CONST_STR("Stack overflow\n"));
        // JIT: pcpu->state = Cpu_Break;
        ir_STORE(ir_ADD_OFFSET(cpu, offsetof(cpu_t, state)), ir_CONST_I32(Cpu_Break));
        ir_RETURN(IR_VOID);
    }

    if (stack_underflow) {
        ir_MERGE_list(stack_underflow);
        // JIT: printf("Stack overflow\n");
        ir_CALL_1(IR_VOID, printf_func, ir_CONST_STR("Stack underflow\n"));
        // JIT: pcpu->state = Cpu_Break;
        ir_STORE(ir_ADD_OFFSET(cpu, offsetof(cpu_t, state)), ir_CONST_I32(Cpu_Break));
        ir_RETURN(IR_VOID);
    }
    if (stack_bound) {
        ir_MERGE_list(stack_bound);
        // JIT: printf("Out of bound picking\n");
        ir_CALL_1(IR_VOID, printf_func, ir_CONST_STR("Stack underflow\n"));
        // JIT: pcpu->state = Cpu_Break;
        ir_STORE(ir_ADD_OFFSET(cpu, offsetof(cpu_t, state)), ir_CONST_I32(Cpu_Break));
        ir_RETURN(IR_VOID);
    }
}

int main(int argc, char **argv) {
    uint64_t steplimit = parse_args(argc, argv);
    cpu_t cpu = init_cpu();
    ir_ctx ctx;
    typedef void (*entry_t)(cpu_t*);
    entry_t entry;
    size_t size;

    ir_init(&ctx, IR_FUNCTION | IR_OPT_FOLDING | IR_OPT_CFG | IR_OPT_CODEGEN, 256, 1024);

    jit_program(&ctx, cpu.pmem, PROGRAM_SIZE);
    ir_save(&ctx, IR_SAVE_CFG | IR_SAVE_RULES | IR_SAVE_REGS, stderr);

    entry = (entry_t)ir_jit_compile(&ctx, 2, &size);
    if (!entry) {
        printf("Compilation failure\n");
    }

    ir_save(&ctx, IR_SAVE_CFG | IR_SAVE_RULES | IR_SAVE_REGS, stderr);
    ir_disasm("prog", entry, size, 0, &ctx, stderr);

    entry(&cpu);

    ir_free(&ctx);

    assert(cpu.state != Cpu_Running || cpu.steps == steplimit);

    /* Print CPU state */
    printf("CPU executed %ld steps. End state \"%s\".\n",
            cpu.steps, cpu.state == Cpu_Halted? "Halted":
                       cpu.state == Cpu_Running? "Running": "Break");
    printf("PC = %#x, SP = %d\n", cpu.pc, cpu.sp);
    printf("Stack: ");
    for (int32_t i=cpu.sp; i >= 0 ; i--) {
        printf("%#10x ", cpu.stack[i]);
    }
    printf("%s\n", cpu.sp == -1? "(empty)": "");

    free(LoadedProgram);

    return cpu.state == Cpu_Halted ||
           (cpu.state == Cpu_Running &&
            cpu.steps == steplimit)?0:1;
}
